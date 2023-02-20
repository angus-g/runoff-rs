use clap::Parser;
use ndarray::s;

#[derive(Parser)]
struct Cli {
    #[arg(short, long)]
    include_box: bool,

    grid_file: String,
    mask_file: String,
    runoff_file: String,
    runoff_area_file: String,
    output_file: String,
}

fn main() -> Result<(), netcdf::error::Error> {
    let cli = Cli::parse();

    let runoff_file = netcdf::open(cli.runoff_file)?;
    let runoff_var = &runoff_file
        .variable("friver")
        .expect("Could not find 'friver' variable");
    let runoff_lon = &runoff_file
        .variable("longitude")
        .expect("Could not find 'longitude' dimension in runoff");
    let runoff_lat = &runoff_file
        .variable("latitude")
        .expect("Could not find 'latitude' dimension in runoff");
    let runoff_time = &runoff_file
	.variable("time")
	.expect("Could not find 'time' dimension in runoff");

    let runoff_nx = runoff_lon.len();
    let runoff_ny = runoff_lat.len();

    let runoff_area_file = netcdf::open(cli.runoff_area_file)?;

    let grid_file = netcdf::open(cli.grid_file)?;
    let mask_file = netcdf::open(cli.mask_file)?;

    let nx = &grid_file
        .dimension("nxp")
        .expect("Could not find 'nxp' dimension in grid")
        .len()
        / 2;
    let ny = &grid_file
        .dimension("nyp")
        .expect("Could not find 'nyp' dimension in grid")
        .len()
        / 2;

    // read the destination tracer points
    // these are in row-major order (C ordering)
    let mut lon_data = ndarray::Array1::<f64>::zeros(nx * ny);
    let mut lat_data = ndarray::Array1::<f64>::zeros(nx * ny);
    let mut mask_data = ndarray::Array1::<f64>::zeros(nx * ny);
    let mut runoff_area_data = ndarray::Array1::<f64>::zeros(runoff_nx * runoff_ny);
    let area_data;
    {
        let lon_data = lon_data.as_slice_mut().unwrap();
        let lon_var = &grid_file
            .variable("x")
            .expect("Could not find 'x' variable in grid");
        lon_var.values_strided_to(lon_data, Some(&[1, 1]), None, &[2, 2])?;

        let lat_data = lat_data.as_slice_mut().unwrap();
        let lat_var = &grid_file
            .variable("y")
            .expect("Could not find 'y' variable in grid");
        lat_var.values_strided_to(lat_data, Some(&[1, 1]), None, &[2, 2])?;

	let mask_data = mask_data.as_slice_mut().unwrap();
	let mask_var = &mask_file
	    .variable("mask")
	    .expect("Could not find 'mask' variable in mask");
	mask_var.values_to(mask_data, None, None)?;

        let mut full_area_data = ndarray::Array2::<f64>::zeros((ny * 2, nx * 2));
        let area_data_buffer = full_area_data.as_slice_mut().unwrap();
        let area_var = &grid_file
            .variable("area")
            .expect("Could not find 'area' variable in grid");
        area_var.values_to(area_data_buffer, None, None)?;

        // decimate by 2x2 blocks
        area_data = ndarray::Array::from_iter(full_area_data.exact_chunks((2, 2)).into_iter().map(|v| v.sum()));

        let runoff_area_data = runoff_area_data.as_slice_mut().unwrap();
        let area_var = &runoff_area_file
            .variable("areacello")
            .expect("Could not find 'areacello' in runoff area file");
        area_var.values_to(runoff_area_data, None, None)?;
    }

    let max_lat = lat_data[[(ny - 1) * nx]];
    println!("max lat: {}", max_lat);
    println!("max lat idx: {}", runoff_lat.values::<f64>(None, None)?.into_iter().filter(|l| l <= &max_lat).count());

    // create a kdtree for the destination grid, including only the non-masked points
    let destination_meshgrid: Vec<(usize, [f64; 2])> = lon_data
        .iter()
        .copied()
        .zip(lat_data.iter().copied())
        .map(|(lon, lat)| [lon, lat])
	.enumerate()
	.zip(mask_data.iter())
	.filter_map(|(coord, mask)| (*mask > 0.0).then_some(coord))
        .collect();
    let kdtree = kd_tree::KdTree2::build_by_key(
	destination_meshgrid,
	|item, k| ordered_float::OrderedFloat(item.1[k]),
    );

    // for each point on the runoff grid, calculate its nearest point,
    // then add the runoff to that cell (multiply by original cell
    // area, divide by new cell area) we want this as a sparse matrix:
    // multiply the runoff field for a given time by this matrix to
    // get the runoff on the destination grid
    let mut source_idx = Vec::new();
    let mut dest_idx = Vec::new();
    let mut scaling = Vec::new();

    // iterate in row-major order to match the other data
    for (j, lat) in runoff_lat.values::<f64>(None, None)?.into_iter().enumerate() {
        if lat > max_lat {
	    continue;
        }

	for (i, lon) in runoff_lon.values::<f64>(None, None)?.iter().enumerate() {
            let lon = if *lon > 80.0 { lon - 360.0 } else { *lon };

            // remap lon onto dest grid range
            let idx = j * runoff_nx + i;

	    if cli.include_box {
		let nearests = kdtree.within_by(
		    &[[lon - 0.125, lat - 0.125], [lon + 0.125, lat + 0.125]],
		    |item, k| item.1[k],
		);
		// spread the runoff evenly across all the matching cells
		let num_cells = nearests.len() as f64;

		for nearest in nearests {
		    let nearest = nearest.0;

		    source_idx.push(idx as isize);
		    dest_idx.push(nearest);
		    let area_scaling = runoff_area_data[idx] / area_data[[nearest]] / num_cells;
		    scaling.push(area_scaling);
		}
	    } else {
		let nearest = kdtree.nearest_by(
		    &[lon, lat],
		    |item, k| item.1[k]
		)
		    .unwrap_or_else(|| panic!("no nearest point for ({},{})", lon, lat));
		let nearest = nearest.item.0;

		source_idx.push(idx as isize);
		dest_idx.push(nearest);

		let area_scaling = runoff_area_data[idx] / area_data[[nearest]];
		scaling.push(area_scaling);
	    }
         }
    }

    let mut runoff_map = rsparse::data::Sprs::new();
    {
	let runoff_trpl = rsparse::data::Trpl{
	    m: ny * nx,
	    n: runoff_nx * runoff_ny,
	    p: source_idx,
	    i: dest_idx,
	    x: scaling,
	};
	runoff_map.from_trpl(&runoff_trpl);
    }

    let mut regrid_file = netcdf::create(cli.output_file)?;
    regrid_file.add_dimension("latitude", ny)?;
    regrid_file.add_dimension("longitude", nx)?;
    regrid_file.add_unlimited_dimension("time")?;

    {
	let mut latitude_var = regrid_file.add_variable::<f64>("latitude", &["latitude"],)?;
	latitude_var.put_values(&lat_data.slice(s![..;nx]).to_vec(), None, None)?;
	latitude_var.add_attribute("units", "degrees_north")?;
	latitude_var.add_attribute("axis", "Y")?;
    }
    {
	let mut longitude_var = regrid_file.add_variable::<f64>("longitude", &["longitude"],)?;
	longitude_var.put_values(lon_data.slice(s![0..nx]).as_slice().unwrap(), None, None)?;
	longitude_var.add_attribute("units", "degrees_east")?;
	longitude_var.add_attribute("axis", "X")?;
    }
    {
	let mut time_var = regrid_file.add_variable::<f64>("time", &["time"],)?;
	time_var.put_values(runoff_time.values::<f64>(None, None)?.as_slice().unwrap(), None, None)?;
	time_var.add_attribute("units", "days since 1900-01-01")?;
	time_var.add_attribute("modulo", " ")?;
	time_var.add_attribute("axis", "T")?;
	time_var.add_attribute("calendar", "noleap")?;
    }

    let mut regrid_var = regrid_file.add_variable::<f64>(
	"friver", &["time", "latitude", "longitude"],
    )?;
    regrid_var.compression(5)?;
    regrid_var.chunking(&[1, 423, 1800])?;
    regrid_var.add_attribute("units", "kg/m2/sec")?;

    // mask off areas out of the domain
    let mut runoff_area = runoff_area_data.into_shape((runoff_ny, runoff_nx)).unwrap();
    runoff_area.slice_mut(s![212.., ..]).fill(0.);

    for time_idx in 0..365 {
	println!("regridding {}", time_idx);

	let runoff_data = runoff_var.values::<f64>(Some(&[time_idx, 0, 0]), Some(&[1, runoff_ny, runoff_nx]))?;
	let total_runoff_before = (&runoff_data * &runoff_area).sum();

	let mut runoff_column = rsparse::data::Sprs::new();
	runoff_column.from_vec(&vec![runoff_data.into_raw_vec()]);
	runoff_column = rsparse::transpose(&runoff_column); // transpose to column vector

	// this will concatenate rows, i.e. row-major
	let runoff_regridded = rsparse::multiply(&runoff_map, &runoff_column).to_dense();
	let runoff_regridded: Vec<f64> = runoff_regridded.iter().flatten().cloned().collect();
	regrid_var.put_values(&runoff_regridded, Some(&[time_idx, 0, 0]), Some(&[1, ny, nx]))?;

	let total_runoff_after = (&area_data * ndarray::Array1::from_vec(runoff_regridded)).sum();
	println!("before: {}, after: {}, diff: {}", total_runoff_before, total_runoff_after, total_runoff_after - total_runoff_before);
	println!("relative: {}", (total_runoff_after - total_runoff_before).abs() / total_runoff_after);
    }

    Ok(())
}
