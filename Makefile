SRC_PATH ?= $(abspath ./peract/utils/voxelize)
CXX := c++
CPPFLAGS := -fPIC $(shell python3-config --includes) \
            -Iextern/pybind11/include \
            $(shell pkg-config --libs Open3D)
CXXFLAGS := -O3 --shared $(shell pkg-config --cflags Open3D)
EXT_SUFFIX := $(shell python3-config --extension-suffix)

all: $(SRC_PATH).cpp
ifeq ($(shell pkg-config --exists Open3D && echo 0), 0)
	$(CXX) -o $(SRC_PATH)${EXT_SUFFIX} $< $(CXXFLAGS) $(CPPFLAGS)
else
	$(error Open3D pkgconfig path must be set in PKG_CONFIG_PATH: \
	https://www.open3d.org/docs/latest/cpp_project.html)
endif

PHONY: .clean
clean:
	rm -f $(SRC_PATH)$(EXT_SUFFIX)
