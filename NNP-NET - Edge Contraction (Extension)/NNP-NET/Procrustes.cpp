#include "Procrustes.h"

#include "pybind11/embed.h"
#include <pybind11/numpy.h>
namespace py = pybind11;
using namespace py::literals;

static double* _inp = nullptr, *_inp2 = nullptr;
static int _rows, _cols;

// PYBIND11_MODULE(getProcrustesInput, m) {
// 	m.def("getI", []() {
// 		return py::array_t<double>({ _rows, _cols }, _inp);
// 		});
// 	m.def("getI2", []() {
// 		return py::array_t<double>({ _rows, _cols }, _inp2);
// 		});
// }

void NNPNet::Procrustes::solve(double* array, double* like, int rows, int cols)
{
	auto l = py::dict(
		"inp1"_a = py::array_t<double>({ rows, cols }, array),
		"inp2"_a = py::array_t<double>({ rows, cols }, like));
	py::exec(R"(
import numpy as np

shift = np.mean(inp2, 0)
scale = np.linalg.norm(inp2 - shift)

_, out, _ = procrustes(inp2, inp1)
out *= scale
out += shift

)", py::globals(), l);
	memcpy(array, (l)["out"].cast<py::array>().data(), sizeof(double)*rows*cols);
}
