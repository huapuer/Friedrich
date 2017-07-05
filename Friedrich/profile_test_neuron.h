#pragma once

#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../Ludwig/Ludwig/ludwig_neural_network.h"
#include "executable.h"

struct map {
	int size;
	float* dev_t;
};

__constant__ map w_mutes;

__global__ void integrate(float* s) {
	/*
	int i = threadIdx.x;
	if (s[i].t > 0.0) {
	s[i].t = 1.0;
	}
	else {
	s[i].t = -1.0;
	}
	*/
}

__global__ void push_forward(float* s) {
	/*
	int i = threadIdx.x;
	if (s[i].t > 0.0) {
		s[i].t = 1.0;
	}
	else {
		s[i].t = -1.0;
	}
	*/
}

__global__ void push_full(float* s, int soffset, float* w, float* r, const int ss, const int ts) {
	for (int i = 0; i < ts; i++) {
		int idx = i*ss + threadIdx.x;
		r[idx] = s[threadIdx.x + soffset] * w[idx];
	}
}

__global__ void mutate(float* w, const unsigned long long gen) {
	/*
	int i = threadIdx.x;
	if (w[i].working_gen == gen) {
	int gap = gen - w[i].gen;
	if (gap == 1) {
	w[i].stage++;
	}
	else {
	w[i].stage = 0;
	}
	if (w[i].stage < w_mutes.size) {
	w[i].t += w_mutes.dev_t[w->stage];
	}
	w[i].gen = gen;
	}
	*/
}

__global__ void clear_pull_forward(const float *s, float *t, int soffset, int toffset, float* w, const int ss, const unsigned long long gen)
{
	int i = threadIdx.x + toffset;
	int j = threadIdx.x + soffset;
	t[i] = 0.0;
	t[i] += s[j] * w[threadIdx.x];
}

__global__ void pull_forward(const float *s, float *t, int soffset, int toffset, float* w, const int ss, const unsigned long long gen)
{
	int i = threadIdx.x + toffset;
	int j = threadIdx.x + soffset;
	t[i] += s[j] * w[threadIdx.x];
}

__global__ void clear_pull_full(float *t, int toffset, float* r, const int ss, const unsigned long long gen)
{
	int i = threadIdx.x + toffset;
	t[i] = 0.0;
	for (int j = 0; j < ss; j++) {
		t[i] += r[threadIdx.x*ss + j];
	}
}

__global__ void pull_full(float *t, int toffset, float* r, const int ss, const unsigned long long gen)
{
	int i = threadIdx.x + toffset;
	for (int j = 0; j < ss; j++) {
		t[i] +=r[threadIdx.x*ss + j];
	}
}

void construct_network() {
	has_layer_phsical(0, 9);

	has_layer_phsical(1, 9);
	has_layer_logical(11, 1, 0, 3, false);
	has_layer_logical(12, 1, 3, 3, false);
	has_layer_logical(13, 1, 6, 3, false);

	has_layer_phsical(21, 3);
	has_layer_logical(210, 21, 0, 3, true);
	has_layer_phsical(22, 3);
	has_layer_logical(220, 22, 0, 3, true);
	has_layer_phsical(23, 3);
	has_layer_logical(230, 23, 0, 3, true);

	has_layer_phsical(3, 6);
	has_layer_logical(30, 3, 0, 6, true);
	has_layer_logical(31, 3, 0, 2, false);
	has_layer_logical(32, 3, 2, 2, false);
	has_layer_logical(33, 3, 4, 2, false);

	has_link(0, LINK_FORWARD, NULL, 0, NULL, 1);

	has_link(1, LINK_FORWARD, NULL, 11, NULL, 21);
	has_link(2, LINK_FORWARD, NULL, 12, NULL, 22);
	has_link(3, LINK_FORWARD, NULL, 13, NULL, 23);

	has_link(4, LINK_FULL, NULL, 21, NULL, 31);
	has_link(5, LINK_FULL, NULL, 22, NULL, 32);
	has_link(6, LINK_FULL, NULL, 23, NULL, 33);

	has_link(11, LINK_FULL, NULL, 31, NULL, 21);
	has_link(12, LINK_FULL, NULL, 32, NULL, 22);
	has_link(13, LINK_FULL, NULL, 33, NULL, 23);

	has_link(7, LINK_FULL, NULL, 21, NULL, 210);
	has_link(8, LINK_FULL, NULL, 22, NULL, 220);
	has_link(9, LINK_FULL, NULL, 23, NULL, 230);

	has_link(10, LINK_FULL, NULL, 3, NULL, 30);
}

void init_network() {
	srand(time(NULL));

	layer_t* next = pick_layer(0);
	while (next) {
		int size = next->size;
		if (size > 0) {
			next->host_t.t = (float*)malloc(sizeof(float)*size);
			//memset(next->t, 0, sizeof(float)*size);
			//TODO: initialize float?
			cudaMalloc((void**)&next->dev_t[0].t, size * sizeof(float));
			cudaMemcpy(next->dev_t[0].t, next->host_t.t, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMalloc((void**)&next->dev_t[1].t, size * sizeof(float));
			cudaMemcpy(next->dev_t[1].t, next->host_t.t, size * sizeof(float), cudaMemcpyHostToDevice);
		}
		next = next->follow;
	}

	link* next_l = pick_link(0);
	while (next_l) {
		int size = next_l->size;

		next_l->host_t.t = (float*)malloc(sizeof(float)*size);
		for (int i = 0; i < size; i++) {
			//if (next_l->id > 0) {
			//	next_l->t[i].t = float(rand()) / float(RAND_MAX) - 0.5f;
			//}
			//else {
				next_l->host_t.t[i] = 1.0f;
			//}
		}
		cudaMalloc((void**)&next_l->dev_t.t, size * sizeof(float));
		cudaMemcpy(next_l->dev_t.t, next_l->host_t.t, size * sizeof(float), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&next_l->dev_t.r, size * sizeof(float));

		next_l = next_l->follow;
	}
}

void external_input(executable** head, executable** tail, unsigned long long gen) {
	layer_t* l = pick_layer(0);

	if (gen == 1) {
		srand(time(NULL));

		int size = l->size;
		for (int i = 0; i < size; i++) {
			//l->t[i].t = float(rand()) / float(RAND_MAX) - 0.5f;
			l->host_t.t[i] = 0.1 + i / (size / 3) * 0.1;
		}
		cudaMemcpy(l->dev_t[l->cur_t_dev_t].t, l->host_t.t, l->size * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(l->dev_t[l->cur_s_dev_t].t, l->host_t.t, l->size * sizeof(float), cudaMemcpyHostToDevice);
	}

	prepend_executable(head, tail, new_executable(gen, EXECUTE_LAYER, l, l->next, l->next->t_layer));
}