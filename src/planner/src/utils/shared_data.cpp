#include "utils/shared_data.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace nmoma_planner::shared_data;
namespace py = pybind11;

PYBIND11_MODULE(shared_data, m)
{
    py::class_<DataPoint>(m, "DataPoint")
    .def(py::init())
    .def_property_readonly("ESDF3D", &DataPoint::getESDF3D)
    .def_property_readonly("traj", &DataPoint::getTraj)
    .def_property_readonly("boxes", &DataPoint::getBoxes)
    .def_property_readonly("box_num", &DataPoint::getBoxNum)
    .def("__repr__", [](const DataPoint& p) { return "<data.DataPoint>"; });


    py::class_<SharedDataProducer>(m, "SharedDataProducer")
    .def(py::init<std::string, unsigned int>())
    .def("produce", &SharedDataProducer::produce);

    py::class_<SharedDataConsumer>(m, "SharedDataConsumer")
    .def(py::init<std::string>())
    .def("consume", &SharedDataConsumer::consume)
    .def("signal_done", &SharedDataConsumer::signal_done)
    .def("getProducerNum", &SharedDataConsumer::getProducerNum);
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
