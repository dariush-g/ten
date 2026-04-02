#include <stdint.h>

void kernel(float **tensors, int n) {
	float *t0 = (float *)tensors[0];
	float *B = (float *)tensors[1];
	float *A = (float *)tensors[2];
	float *t1 = (float *)tensors[3];
	float *bias = (float *)tensors[4];
	float *t2 = (float *)tensors[5];

	for (int i = 0; i < 1; i++) {
		for (int j = 0; j < 1; j++) {
			for (int k = 0; k < 2; k++) {
				t0[i + j] += A[i * 2 + k] * B[k + j];
			}
		}
	}

	for (int i = 0; i < 1; i++) {
		for (int j = 0; j < 1; j++) {
			t1[i + j] = t0[i + j] + bias[j];
		}
	}

	for (int i = 0; i < 1; i++) {
		for (int j = 0; j < 1; j++) {
			t2[i + j] = t1[i + j] > 0.0f ? t1[i + j] : 0.0f;
		}
	}
}