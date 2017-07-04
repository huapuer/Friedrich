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
		t[i] += r[threadIdx.x*ss + j];
	}
}

void construct_network() {
	has_layer_phsical(0, 27);

	has_layer_phsical(1, 27);
	has_layer_logical(11, 1, 0, 3, false);
	has_layer_logical(12, 1, 3, 3, false);
	has_layer_logical(13, 1, 6, 3, false);
	has_layer_logical(14, 1, 9, 3, false);
	has_layer_logical(15, 1, 12, 3, false);
	has_layer_logical(16, 1, 15, 3, false);
	has_layer_logical(17, 1, 18, 3, false);
	has_layer_logical(18, 1, 21, 3, false);
	has_layer_logical(19, 1, 24, 3, false);

	has_layer_phsical(21, 3);
	has_layer_logical(210, 21, 0, 3, true);
	has_layer_phsical(22, 3);
	has_layer_logical(220, 22, 0, 3, true);
	has_layer_phsical(23, 3);
	has_layer_logical(230, 23, 0, 3, true);
	has_layer_phsical(24, 3);
	has_layer_logical(240, 24, 0, 3, true);
	has_layer_phsical(25, 3);
	has_layer_logical(250, 25, 0, 3, true);
	has_layer_phsical(26, 3);
	has_layer_logical(260, 26, 0, 3, true);
	has_layer_phsical(27, 3);
	has_layer_logical(270, 27, 0, 3, true);
	has_layer_phsical(28, 3);
	has_layer_logical(280, 28, 0, 3, true);
	has_layer_phsical(29, 3);
	has_layer_logical(290, 29, 0, 3, true);

	has_layer_phsical(31, 6);
	has_layer_logical(310, 31, 0, 6, true);
	has_layer_logical(311, 31, 0, 2, false);
	has_layer_logical(312, 31, 2, 2, false);
	has_layer_logical(313, 31, 4, 2, false);

	has_layer_phsical(32, 6);
	has_layer_logical(320, 32, 0, 6, true);
	has_layer_logical(321, 32, 0, 2, false);
	has_layer_logical(322, 32, 2, 2, false);
	has_layer_logical(323, 32, 4, 2, false);

	has_layer_phsical(33, 6);
	has_layer_logical(330, 33, 0, 6, true);
	has_layer_logical(331, 33, 0, 2, false);
	has_layer_logical(332, 33, 2, 2, false);
	has_layer_logical(333, 33, 4, 2, false);

	has_layer_phsical(4, 18);
	has_layer_logical(40, 4, 0, 18, true);
	has_layer_logical(41, 4, 0, 6, false);
	has_layer_logical(42, 4, 6, 6, false);
	has_layer_logical(43, 4, 12, 6, false);


	has_link(0, LINK_FORWARD, NULL, 0, NULL, 1);

	has_link(1, LINK_FORWARD, NULL, 11, NULL, 21);
	has_link(2, LINK_FORWARD, NULL, 12, NULL, 22);
	has_link(3, LINK_FORWARD, NULL, 13, NULL, 23);
	has_link(4, LINK_FORWARD, NULL, 14, NULL, 24);
	has_link(5, LINK_FORWARD, NULL, 15, NULL, 25);
	has_link(6, LINK_FORWARD, NULL, 16, NULL, 26);
	has_link(7, LINK_FORWARD, NULL, 17, NULL, 27);
	has_link(8, LINK_FORWARD, NULL, 18, NULL, 28);
	has_link(9, LINK_FORWARD, NULL, 19, NULL, 29);

	has_link(10, LINK_FULL, NULL, 21, NULL, 311);
	has_link(11, LINK_FULL, NULL, 22, NULL, 312);
	has_link(12, LINK_FULL, NULL, 23, NULL, 313);
	has_link(13, LINK_FULL, NULL, 24, NULL, 321);
	has_link(14, LINK_FULL, NULL, 25, NULL, 322);
	has_link(15, LINK_FULL, NULL, 26, NULL, 323);
	has_link(16, LINK_FULL, NULL, 27, NULL, 331);
	has_link(17, LINK_FULL, NULL, 28, NULL, 332);
	has_link(18, LINK_FULL, NULL, 29, NULL, 333);

	has_link(19, LINK_FULL, NULL, 311, NULL, 21);
	has_link(20, LINK_FULL, NULL, 312, NULL, 22);
	has_link(21, LINK_FULL, NULL, 313, NULL, 23);
	has_link(22, LINK_FULL, NULL, 321, NULL, 24);
	has_link(23, LINK_FULL, NULL, 322, NULL, 25);
	has_link(24, LINK_FULL, NULL, 323, NULL, 26);
	has_link(25, LINK_FULL, NULL, 331, NULL, 27);
	has_link(26, LINK_FULL, NULL, 332, NULL, 28);
	has_link(27, LINK_FULL, NULL, 333, NULL, 29);

	has_link(28, LINK_FULL, NULL, 31, NULL, 41);
	has_link(29, LINK_FULL, NULL, 32, NULL, 42);
	has_link(30, LINK_FULL, NULL, 33, NULL, 43);

	has_link(31, LINK_FULL, NULL, 41, NULL, 31);
	has_link(32, LINK_FULL, NULL, 42, NULL, 32);
	has_link(33, LINK_FULL, NULL, 43, NULL, 33);

	has_link(34, LINK_FULL, NULL, 21, NULL, 210);
	has_link(35, LINK_FULL, NULL, 22, NULL, 220);
	has_link(36, LINK_FULL, NULL, 23, NULL, 230);
	has_link(37, LINK_FULL, NULL, 24, NULL, 240);
	has_link(38, LINK_FULL, NULL, 25, NULL, 250);
	has_link(39, LINK_FULL, NULL, 26, NULL, 260);
	has_link(40, LINK_FULL, NULL, 27, NULL, 270);
	has_link(41, LINK_FULL, NULL, 28, NULL, 280);
	has_link(42, LINK_FULL, NULL, 29, NULL, 290);

	has_link(43, LINK_FULL, NULL, 31, NULL, 310);
	has_link(44, LINK_FULL, NULL, 32, NULL, 320);
	has_link(45, LINK_FULL, NULL, 33, NULL, 330);

	has_link(46, LINK_FULL, NULL, 4, NULL, 40);
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