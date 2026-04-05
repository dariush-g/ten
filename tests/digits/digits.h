#pragma once

#include "../../src/ten.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

inline std::vector<float> load_floats(const std::string &path, int count) {
	std::vector<float> data(count);
	std::ifstream f(path, std::ios::binary);
	if (!f.is_open())
		throw std::runtime_error("failed to open: " + path);
	f.read(reinterpret_cast<char *>(data.data()), count * sizeof(float));
	return data;
}
inline void test_digits_inference() {
	auto w1 = load_floats("./tests/digits/weights/0.weight.bin", 128 * 784);
	auto b1 = load_floats("./tests/digits/weights/0.bias.bin", 128);
	auto w2 = load_floats("./tests/digits/weights/2.weight.bin", 10 * 128);
	auto b2 = load_floats("./tests/digits/weights/2.bias.bin", 10);

	auto images =
		load_floats("./tests/digits/weights/test_images.bin", 10000 * 784);
	auto labels = load_floats("./tests/digits/weights/test_labels.bin", 10000);

	auto input = ten::f32({1, 784}, "input");
	auto W1 = ten::f32({784, 128}, "W1");
	auto B1 = ten::f32({128}, "b1");
	auto W2 = ten::f32({128, 10}, "W2");
	auto B2 = ten::f32({10}, "b2");

	ten::Builder b;
	auto h = b.matmul(input, W1);
	auto h2 = b.bias_add(h, B1);
	auto h3 = b.relu(h2);
	auto o = b.matmul(h3, W2);
	auto o2 = b.bias_add(o, B2);

	auto kernel = b.compile();

	// debug: print first 10 predictions
	for (int i = 0; i < 10; i++) {
		float *img = images.data() + i * 784;
		kernel({
			{input, img},
			{W1, w1.data()},
			{B1, b1.data()},
			{W2, w2.data()},
			{B2, b2.data()},
		});

		auto [data, shape] = kernel.get(o2);
		int pred = 0;
		for (int j = 1; j < 10; j++)
			if (data[j] > data[pred])
				pred = j;
	}

	// full accuracy
	int correct = 0;
	for (int i = 0; i < 10000; i++) {
		float *img = images.data() + i * 784;
		kernel({
			{input, img},
			{W1, w1.data()},
			{B1, b1.data()},
			{W2, w2.data()},
			{B2, b2.data()},
		});

		auto [data, shape] = kernel.get(o2);
		int pred = 0;
		for (int j = 1; j < 10; j++)
			if (data[j] > data[pred])
				pred = j;

		if (pred == static_cast<int>(labels[i]))
			correct++;
	}

	std::cout << "accuracy: " << correct << "/10000\n";
}