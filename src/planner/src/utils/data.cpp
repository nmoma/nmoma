#include "utils/data.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace nmoma_planner::data;
namespace py = pybind11;

PYBIND11_MODULE(data, m)
{
    py::class_<DataPoint<10>>(m, "DataPoint")
    .def(py::init())
    .def(py::init<std::string>())
    // .def_property_readonly("occ3D", &DataPoint<10>::getOcc3D)
    .def_property_readonly("ESDF3D", &DataPoint<10>::getESDF3D)
    .def_property_readonly("traj", &DataPoint<10>::getTraj)
    .def_property_readonly("boxes", &DataPoint<10>::getBoxes)
    .def_property_readonly("hash", &DataPoint<10>::getHash)
    .def("__repr__", [](const DataPoint<10>& p) { return "<data.DataPoint>"; });
    
    py::class_<DataLoader<10>>(m, "DataLoader")
    .def(py::init<std::string>())
    .def("hasNext", &DataLoader<10>::has_next)
    .def("next", &DataLoader<10>::next)
    .def("size", &DataLoader<10>::size);

    m.def("load_binary_datapoints", &load_binary_datapoints<10>);
    m.def("hash_value", &hash_value<10>);
}

// Override for text_oarchive
// namespace boost { 
// namespace serialization {
//     template <int dof>
//     struct version<DataPoint<dof>> {
//         // static constexpr unsigned int value = 2;
//         BOOST_STATIC_CONSTANT(unsigned int, value = 2);
//     };
// }}
