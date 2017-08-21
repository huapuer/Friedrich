#pragma once

#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../Ludwig/Ludwig/ludwig_neural_network.h"
#include "../../Ludwig/Ludwig/ludwig_net_sync.h"
#include "executable.h"
#include <math.h>

struct map {
	int size;
	float* dev_t;
};

__constant__ map w_mutes;


__global__ void joint(float* t, int toffset, float* r2, float* po,float* po_sum, bool out, float* out_a, float* out_b) {
	int i = threadIdx.x + toffset;

	float diff = t[i] - r2[i];
	po[i] += diff*diff;
	po_sum[i] += t[i] * t[i];
	r2[i] = t[i];

	if (out) {
		out_b[i] = po[i];
		out_a[i] = po_sum[i];
	}
}

__global__ void integrate(float* s, float* lmbd) {
	int i = threadIdx.x;

	s[i] *= 1.0 / (1.0 + exp(- (s[i] - 5.0) * 0.1)) * lmbd[i];	//sigmoid, lmbd = +1 or -1
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

__global__ void push_full(float* s, int soffset, float* r, float* pr, float* pr_sum, const int ss, const int ts) {
	for (int i = 0; i < ts; i++) {
		int idx = i*ss + threadIdx.x;

		float diff = s[threadIdx.x + soffset] - r[idx];
		pr[idx] += diff * diff;
		pr_sum[idx] += s[threadIdx.x + soffset] * s[threadIdx.x + soffset];
		r[idx] = s[threadIdx.x + soffset];
	}
}

__global__ void mutate(float* w, float* pr, float* po, float* pr_sum, float* po_sum, const unsigned long long gen) {
	int i = threadIdx.x;
	w[i] *= 1.0 + (po[i] / po_sum[i] - pr[i] /pr_sum[i]) / 2.0;		//0 < p* / p*_sum < 1
	pr[i] = 0;
	po[i] = 0;
	pr_sum[i] = 0;
	po_sum[i] = 0;
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

__global__ void clear_pull_full(float *t, int toffset, float* r, float* w, const int ss, const unsigned long long gen)
{
	int i = threadIdx.x + toffset;
	t[i] = 0.0;
	for (int j = 0; j < ss; j++) {
		t[i] += r[threadIdx.x*ss + j] * w[threadIdx.x*ss + j];
	}
}

__global__ void pull_full(float *t, int toffset, float* r, float* w, const int ss, const unsigned long long gen)
{
	int i = threadIdx.x + toffset;
	for (int j = 0; j < ss; j++) {
		t[i] += r[threadIdx.x*ss + j] * w[threadIdx.x*ss + j];
	}
}

void construct_network() {
	has_layer_phsical(0, 10240, false);
	has_layer_logical(1, 0, 0, 10240, true);

	has_link(0, LINK_FULL, NULL, 0, NULL, 1);
}

void external_input(char* content, int size) {
	layer_t* l = pick_layer(0);

	cudaMemcpy(l->dev_t[l->cur_t_dev_t].t + 1024* sizeof(float), content, l->size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(l->dev_t[l->cur_s_dev_t].t + 1024 * sizeof(float), content, l->size * sizeof(float), cudaMemcpyHostToDevice);
}

void init_network() {

	friedrich_acts(net_events::EVENT_STATE, external_input);
	friedrich_talking(9999);

	srand(time(NULL));

	layer_t* next = pick_layer(0);
	while (next) {
		int size = next->size;
		if (size > 0) {
			next->host_t.t = (float*)malloc(sizeof(float)*size);
			cudaMalloc((void**)&next->dev_t[0].t, size * sizeof(float));
			cudaMemcpy(next->dev_t[0].t, next->host_t.t, size * sizeof(float), cudaMemcpyHostToDevice);
			cudaMalloc((void**)&next->dev_t[1].t, size * sizeof(float));
			cudaMemcpy(next->dev_t[1].t, next->host_t.t, size * sizeof(float), cudaMemcpyHostToDevice);

			cudaMalloc((void**)&next->lmbd.t, size * sizeof(float));
			for (int i = 0; i < size; i++) {
				if (i < size / 3) {
					next->host_t.t[i] = -1.0;
				}
				else {
					next->host_t.t[i] = 1.0;
				}
			}
			cudaMemcpy(next->lmbd.t, next->host_t.t, size * sizeof(float), cudaMemcpyHostToDevice);
			
		}
		next = next->follow;
	}

	link* next_l = pick_link(0);
	while (next_l) {
		int size = next_l->size;

		next_l->host_t.t = (float*)malloc(sizeof(float)*size);
		for (int i = 0; i < size; i++) {
			if (next_l->id > 0) {
				next_l->host_t.t[i] = float(rand()) / float(RAND_MAX);
			}
			else {
				next_l->host_t.t[i] = 1.0f;
			}
		}
		cudaMalloc((void**)&next_l->dev_t.t, size * sizeof(float));
		cudaMemcpy(next_l->dev_t.t, next_l->host_t.t, size * sizeof(float), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&next_l->dev_t.r, size * sizeof(float));
		cudaMalloc((void**)&next_l->dev_t.r2, size * sizeof(float));
		cudaMalloc((void**)&next_l->dev_t.pr, size * sizeof(float));
		cudaMalloc((void**)&next_l->dev_t.po, size * sizeof(float));
		cudaMalloc((void**)&next_l->dev_t.pr_sum, size * sizeof(float));
		cudaMalloc((void**)&next_l->dev_t.po_sum, size * sizeof(float));

		next_l = next_l->follow;
	}
}

float* out_b = nullptr;
float* out_a = nullptr;
unsigned long code = 0;

void external(executable** head, executable** tail, unsigned long long gen) {
	if (gen % 10 == 0) {
		layer_t* out = pick_layer(0);

		if (out_b == nullptr) {
			out_b = (float*)malloc(sizeof(float)*out->size);
		}

		if (out_a == nullptr) {
			out_a = (float*)malloc(sizeof(float)*out->size);
		}

		cudaMemcpy(out_b, out->out_b.t + 1024 * sizeof(float), out->size * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(out_a, out->out_a.t + 1024 * sizeof(float), out->size * sizeof(float), cudaMemcpyDeviceToHost);

		/*TO FIX:*/
		for (int i = 0; i < out->size; i++) {
			out_b[i] *= out_b[i];
			if (out_b[i] / out_a[i] > 0.111) {
				if (out_a[i] > 0.555) {
					code += (1 << i);
				}
			}
			else {
				code = float(rand()) / float(RAND_MAX) * 4.0;
				break;
			}
		}

		friedrich_says(net_events(code), (char*)&code, sizeof(unsigned long));

		friedrich_hearing();

		layer_t* in = pick_layer(0);
		prepend_executable(head, tail, new_executable(gen, EXECUTE_LAYER, in, in->next, in->next->t_layer));
	}
}

bool do_mutate(const unsigned long long gen) {
	return gen % 50 == 0;
}