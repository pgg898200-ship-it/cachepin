#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "cutstream.hpp"

#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using SharedPlan = std::shared_ptr<const cutstream::Plan>;
constexpr const char* plan_name = "cutstream.Plan";
constexpr const char* stream_name = "cutstream.Stream";

struct Stream {
    SharedPlan plan;
    cutstream::Cursor cursor{};
    size_t consumed = 0;
    int match = -1;
};

void drop_plan(PyObject* capsule) {
    delete static_cast<SharedPlan*>(PyCapsule_GetPointer(capsule, plan_name));
}
void drop_stream(PyObject* capsule) {
    delete static_cast<Stream*>(PyCapsule_GetPointer(capsule, stream_name));
}
Stream* get_stream(PyObject* obj) {
    return static_cast<Stream*>(PyCapsule_GetPointer(obj, stream_name));
}
PyObject* compile_plan(PyObject*, PyObject* items) {
    if (!PyTuple_Check(items)) {
        PyErr_SetString(PyExc_TypeError, "expected a tuple of byte strings");
        return nullptr;
    }
    try {
        std::vector<std::string> markers;
        for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(items); ++i) {
            auto* b = PyTuple_GET_ITEM(items, i);
            if (!PyBytes_Check(b)) {
                PyErr_SetString(PyExc_TypeError, "each marker must be bytes");
                return nullptr;
            }
            markers.emplace_back(PyBytes_AS_STRING(b), PyBytes_GET_SIZE(b));
        }
        auto ptr = std::make_unique<SharedPlan>(std::make_shared<cutstream::Plan>(std::move(markers)));
        PyObject* cap = PyCapsule_New(ptr.get(), plan_name, drop_plan);
        if (cap) ptr.release();
        return cap;
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::exception& err) {
        PyErr_SetString(PyExc_ValueError, err.what());
        return nullptr;
    }
}
PyObject* new_stream(PyObject*, PyObject* arg) {
    auto* plan = static_cast<SharedPlan*>(PyCapsule_GetPointer(arg, plan_name));
    if (!plan) return nullptr;
    try {
        auto ptr = std::make_unique<Stream>();
        ptr->plan = *plan;
        PyObject* cap = PyCapsule_New(ptr.get(), stream_name, drop_stream);
        if (cap) ptr.release();
        return cap;
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    }
}
PyObject* feed(PyObject*, PyObject* args) {
    PyObject* capsule;
    PyObject* chunk;
    if (!PyArg_ParseTuple(args, "OO!:feed", &capsule, &PyBytes_Type, &chunk)) return nullptr;
    auto* stream = get_stream(capsule);
    if (!stream) return nullptr;
    const auto bytes = std::string_view(PyBytes_AS_STRING(chunk), PyBytes_GET_SIZE(chunk));
    auto next = stream->cursor;
    const auto out = stream->plan->feed(next, bytes);
    if (out.prefix.size() > static_cast<size_t>(PY_SSIZE_T_MAX) - out.input.size())
        return PyErr_NoMemory();
    PyObject* result;
    if (out.prefix.empty() && out.input.size() == bytes.size()) {
        result = Py_NewRef(chunk);
    } else {
        result = PyBytes_FromStringAndSize(nullptr, out.prefix.size() + out.input.size());
        if (!result) return nullptr;
        char* dst = PyBytes_AS_STRING(result);
        if (!out.prefix.empty()) std::memcpy(dst, out.prefix.data(), out.prefix.size());
        if (!out.input.empty()) std::memcpy(dst + out.prefix.size(), out.input.data(), out.input.size());
    }
    stream->cursor = next;
    stream->consumed = out.consumed;
    if (out.match >= 0) stream->match = out.match;
    return result;
}
PyObject* finish(PyObject*, PyObject* arg) {
    auto* stream = get_stream(arg);
    if (!stream) return nullptr;
    auto next = stream->cursor;
    const auto tail = stream->plan->finish(next);
    PyObject* result = PyBytes_FromStringAndSize(tail.data(), tail.size());
    if (result) stream->cursor = next;
    return result;
}
PyObject* pending(PyObject*, PyObject* arg) {
    auto* stream = get_stream(arg);
    if (!stream) return nullptr;
    const auto tail = stream->plan->pending(stream->cursor);
    return PyBytes_FromStringAndSize(tail.data(), tail.size());
}
PyObject* done(PyObject*, PyObject* arg) {
    auto* stream = get_stream(arg);
    return stream ? PyBool_FromLong(stream->cursor.done()) : nullptr;
}
PyObject* match(PyObject*, PyObject* arg) {
    auto* stream = get_stream(arg);
    if (!stream) return nullptr;
    if (stream->match < 0) Py_RETURN_NONE;
    return PyLong_FromLong(stream->match);
}
PyObject* consumed(PyObject*, PyObject* arg) {
    auto* stream = get_stream(arg);
    return stream ? PyLong_FromSize_t(stream->consumed) : nullptr;
}
PyMethodDef methods[] = {
    {"compile", compile_plan, METH_O, nullptr},
    {"stream", new_stream, METH_O, nullptr},
    {"feed", feed, METH_VARARGS, nullptr},
    {"finish", finish, METH_O, nullptr},
    {"pending", pending, METH_O, nullptr},
    {"done", done, METH_O, nullptr},
    {"match", match, METH_O, nullptr},
    {"consumed", consumed, METH_O, nullptr},
    {nullptr, nullptr, 0, nullptr}
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_native", nullptr, -1, methods};
}
PyMODINIT_FUNC PyInit__native() { return PyModule_Create(&module); }
