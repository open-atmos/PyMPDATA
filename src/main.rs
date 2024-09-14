use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyTuple};
use std::io::*;
use std::fs::*;

fn main() -> PyResult<()> {
    Python::with_gil(|py| {
        //imports
        let vector_field = py.import_bound("PyMPDATA")?.getattr("VectorField")?;
        let scalar_field = py.import_bound("PyMPDATA")?.getattr("ScalarField")?;
        let periodic = py.import_bound("PyMPDATA.boundary_conditions")?.getattr("Periodic")?;
        let stepper_ = py.import_bound("PyMPDATA")?.getattr("Stepper")?;
        let solver_ = py.import_bound("PyMPDATA")?.getattr("Solver")?;
        //set options
        let options_args = [("n_iters", 2)].into_py_dict_bound(py);
        let options = py.import_bound("PyMPDATA")?.getattr("Options")?.call((), Some(&options_args))?;
        //set conditions
        let nx_ny = [24, 24];
        let Cx_Cy = [-0.5, -0.25];
        let boundary_con = PyTuple::new_bound(py, [periodic.call0()?, periodic.call0()?]).into_any();
        let halo = options.getattr("n_halo")?;
        //because rust doesn't like using math symbols in function calls, we pass "indices" to python directly as a string, later doing the same with np.full
        let indices = PyDict::new_bound(py);
        Python::run_bound(py, &format!(r#"import numpy as np
nx, ny = {}, {}
xi, yi = np.indices((nx, ny), dtype=float)
data=np.exp(
-(xi+.5-nx/2)**2 / (2*(ny/10)**2)
-(yi+.5-nx/2)**2 / (2*(ny/10)**2)
)"#, nx_ny[0], nx_ny[1]), None, Some(&indices)).unwrap();
        //when calling python functions from rust, we must use PyDict type for kwagrs
        let advectee_arg = vec![("data", indices.get_item("data")?), ("halo", Some(halo.clone())), ("boundary_conditions", Some(boundary_con))].into_py_dict_bound(py);
        let advectee = scalar_field.call((), Some(&advectee_arg))?;
        let full = PyDict::new_bound(py);
        Python::run_bound(py, &format!(r#"import numpy as np
nx, ny = {}, {}
Cx, Cy = {}, {}
data = (np.full((nx + 1, ny), Cx), np.full((nx, ny + 1), Cy))"#, nx_ny[0], nx_ny[1], Cx_Cy[0], Cx_Cy[1]), None, Some(&full)).unwrap();
        let boundary_con = PyTuple::new_bound(py, [periodic.call0()?, periodic.call0()?]).into_any();
        let advector_arg = vec![("data", full.get_item("data")?), ("halo", Some(halo.clone())), ("boundary_conditions", Some(boundary_con))].into_py_dict_bound(py);
        let advector = vector_field.call((), Some(&advector_arg))?;
        let stepper_arg = vec![("options", options), ("grid", PyTuple::new_bound(py, nx_ny).into_any())].into_py_dict_bound(py);
        // or alternatively let stepper_arg = vec![("options", options),("n_dims", 2)].into_py_dict_bound(py)
        let stepper = stepper_.call((), Some(&stepper_arg))?;
        let mut solver = solver_.call((), Some(&vec![("stepper", stepper), ("advectee", advectee), ("advector", advector)].into_py_dict_bound(py)))?;
        let state_0 = solver.getattr("advectee")?.getattr("get")?.call0()?.getattr("copy")?.call0()?;
        solver.getattr("advance")?.call((), Some(&vec![("n_steps", 75)].into_py_dict_bound(py)))?;
        let state = solver.getattr("advectee")?.getattr("get")?.call0()?;
        Ok(())
    })
}