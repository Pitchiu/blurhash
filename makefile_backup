TARGET := blurhash

BUILD_DIR := ./build
SRC_DIRS := ./src

FLAGS := -std=c++11
CXXFLAGS := $(FLAGS)

CXX := nvcc

SRCS := $(shell find $(SRC_DIRS) -name '*.cu')

OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)

INC_DIRS := $(shell find $(SRC_DIRS) -type d)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.cu.o: %.cu
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -r $(BUILD_DIR)
	rm $(TARGET)