#pragma once

#include "stdfx.h"
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

#define NEURON_NUM 344
#define NEGITIVE_NEURON_CUT 3
#define INPUT_NEURON_NUM 256

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

__global__ void integrate(float* s, float* lmbd, const int ss) {
	int i = threadIdx.x;

	s[i] = 1.0 / (1.0 + exp(- (s[i] - 5.0) * 0.1)) * lmbd[i];	//sigmoid, lmbd = +1 or -1

#ifdef DEBUG_KERNEL_INTEGRATE
	__syncthreads();

	if (threadIdx.x == 0) {
		printf("KERNEL: integrate: r:\n");
		for (int i = 0; i < NEURON_NUM; i++) {
			printf("%12.11f\n", s[i]);
		}
	}
#endif
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

#ifdef DEBUG_KERNEL_PUSH_FULL
	__syncthreads();

	if (threadIdx.x == 0) {
		printf("KERNEL: push_full: r:\n");
		for (int i = 0; i < NEURON_NUM * NEURON_NUM; i++) {
			printf("%f\n", r[i]);
		}
	}
#endif
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

#ifdef DEBUG_KERNEL_CLEAR_PULL_FULL
	__syncthreads();

	if (threadIdx.x == 0) {
		printf("KERNEL: clear_pull_full: t:\n");
		for (int i = 0; i < NEURON_NUM; i++) {
			printf("%f\n", t[i]);
		}
	}
#endif
}

__global__ void pull_full(float *t, int toffset, float* r, float* w, const int ss, const unsigned long long gen)
{
	int i = threadIdx.x + toffset;
	for (int j = 0; j < ss; j++) {
		t[i] += r[threadIdx.x*ss + j] * w[threadIdx.x*ss + j];
	}
}

void construct_network() {
	has_layer_phsical(0, NEURON_NUM, true);
	has_layer_logical(1, 0, 0, NEURON_NUM, false);

	has_link(0, LINK_FULL, NULL, 0, NULL, 1);
}

float* buffer = nullptr;
void external_input(char* content, int size) {
	if (size != INPUT_NEURON_NUM) {
		ERROR("Illegal input!");
		exit(1);
	}

	if (buffer == nullptr) {
		buffer = (float*)malloc(size * sizeof(float));
	}
	for (int i = 0; i < size; i++) {
		if (content[i] > 0) {
			buffer[i] = 1.0f;
		}
		else {
			buffer[i] = 0.0f;
		}
	}

#ifdef DEBUG_EXTERNAL
	fprintf(stdout, "INPUT FORM REMOTE:\n");
	for (int i = 0; i < 32; i++) {
		for (int j = 0; j < 8; j++) {
			fprintf(stdout, "%d", content[i * 8 + j]);
			if (j == 7) {
				fprintf(stdout, "\n");
			}
		}
	}
#endif

	layer_t* l = pick_layer(0);
	cudaMemcpy(l->dev_t[l->cur_t_dev_t].t + NEURON_NUM / NEGITIVE_NEURON_CUT * sizeof(float), buffer, size * sizeof(float), cudaMemcpyHostToDevice);

#ifdef DEBUG_EXTERNAL
	cudaMemcpy(buffer, l->dev_t[l->cur_t_dev_t].t + NEURON_NUM / NEGITIVE_NEURON_CUT * sizeof(float), size * sizeof(float), cudaMemcpyDeviceToHost);
	fprintf(stdout, "RECEIVED INPUT:\n");
	for (int i = 0; i < 32; i++) {
		for (int j = 0; j < 8; j++) {
			fprintf(stdout, "%f", buffer[i * 8 + j]);
			if (j == 7) {
				fprintf(stdout, "\n");
			}
		}
	}
#endif
}

void init_network() {

	friedrich_acts(net_events::EVENT_STATE, external_input);
	friedrich_talking(9999);

	srand(time(NULL));

	layer_t* next = pick_layer(0);
	while (next) {
		if (next->type == LAYER_PHSICAL) {
			int size = next->size;
			if (size > 0) {
				next->host_t.t = (float*)malloc(sizeof(float)*size);
				cudaError_t err = cudaMalloc((void**)&next->dev_t[0].t, size * sizeof(float));
				if (err > 0) {
					ERROR("CUDA MEMORY MALLOC ERROR!");
				}
				err = cudaMemcpy(next->dev_t[0].t, next->host_t.t, size * sizeof(float), cudaMemcpyHostToDevice);
				if (err > 0) {
					ERROR("CUDA MEMORY COPY ERROR!");
				}
				err = cudaMalloc((void**)&next->dev_t[1].t, size * sizeof(float));
				if (err > 0) {
					ERROR("CUDA MEMORY MALLOC ERROR!");
				}
				err = cudaMemcpy(next->dev_t[1].t, next->host_t.t, size * sizeof(float), cudaMemcpyHostToDevice);
				if (err > 0) {
					ERROR("CUDA MEMORY COPY ERROR!");
				}
				err = cudaMalloc((void**)&next->out_a.t, size * sizeof(float));
				if (err > 0) {
					ERROR("CUDA MEMORY MALLOC ERROR!");
				}
				err = cudaMemcpy(next->out_a.t, next->host_t.t, size * sizeof(float), cudaMemcpyHostToDevice);
				if (err > 0) {
					ERROR("CUDA MEMORY COPY ERROR!");
				}
				err = cudaMalloc((void**)&next->out_b.t, size * sizeof(float));
				if (err > 0) {
					ERROR("CUDA MEMORY MALLOC ERROR!");
				}
				err = cudaMemcpy(next->out_b.t, next->host_t.t, size * sizeof(float), cudaMemcpyHostToDevice);
				if (err > 0) {
					ERROR("CUDA MEMORY COPY ERROR!");
				}

				err = cudaMalloc((void**)&next->lmbd.t, size * sizeof(float));
				if (err > 0) {
					ERROR("CUDA MEMORY MALLOC ERROR!");
				}
				for (int i = 0; i < size; i++) {
					if (i < size / NEGITIVE_NEURON_CUT) {
						next->host_t.t[i] = -1.0;
					}
					else {
						next->host_t.t[i] = 1.0;
					}
				}
				err = cudaMemcpy(next->lmbd.t, next->host_t.t, size * sizeof(float), cudaMemcpyHostToDevice);
				if (err > 0) {
					ERROR("CUDA MEMORY COPY ERROR!");
				}
#ifdef DEBUG_INIT_LAYER
				cudaMemcpy(next->host_t.t, next->lmbd.t, size * sizeof(float), cudaMemcpyDeviceToHost);
				fprintf(stdout, "LAMBDA INITED:\n");
				for (int i = 0; i < size; i++) {
					fprintf(stdout, "%f", next->host_t.t[i]);
				}
				fprintf(stdout, "\n");
#endif
			}
		}
		next = next->follow;
	}

	link* next_l = pick_link(0);
	while (next_l) {
		int size = next_l->size;

		next_l->host_t.t = (float*)malloc(sizeof(float)*size);
		int edgeSize = sqrtl(size);
		if (pow(edgeSize, 2) != size) {
			ERROR("ILLEGAL SIZE OF LINK!");
		}
		for (int i = 0; i < edgeSize; i++) {
			for (int j = 0; j < edgeSize; j++) {
				if (i != j) {
					next_l->host_t.t[i * edgeSize + j] = float(rand()) / float(RAND_MAX);
				}
				else {
					next_l->host_t.t[i * edgeSize + j] = 0.0f;
				}
			}
		}

		cudaError_t err = cudaMalloc((void**)&next_l->dev_t.t, size * sizeof(float));
		if (err > 0) {
			ERROR("CUDA MEMORY MALLOC ERROR!");
		}
		err = cudaMemcpy(next_l->dev_t.t, next_l->host_t.t, size * sizeof(float), cudaMemcpyHostToDevice);
		if (err > 0) {
			ERROR("CUDA MEMORY COPY ERROR!");
		}
#ifdef DEBUG_INIT_LINK
		cudaMemcpy(next_l->host_t.t, next_l->dev_t.t, size * sizeof(float), cudaMemcpyDeviceToHost);
		fprintf(stdout, "WEIGHTS INITED:\n");
		for (int i = 0; i < size; i++) {
			fprintf(stdout, "%f\n", next_l->host_t.t[i]);
		}
		fprintf(stdout, "\n");
#endif

		err = cudaMalloc((void**)&next_l->dev_t.r, size * sizeof(float));
		if (err > 0) {
			ERROR("CUDA MEMORY MALLOC ERROR!");
		}
		err = cudaMalloc((void**)&next_l->dev_t.r2, size * sizeof(float));
		if (err > 0) {
			ERROR("CUDA MEMORY MALLOC ERROR!");
		}
		err = cudaMalloc((void**)&next_l->dev_t.pr, size * sizeof(float));
		if (err > 0) {
			ERROR("CUDA MEMORY MALLOC ERROR!");
		}
		err = cudaMalloc((void**)&next_l->dev_t.po, size * sizeof(float));
		if (err > 0) {
			ERROR("CUDA MEMORY MALLOC ERROR!");
		}
		err = cudaMalloc((void**)&next_l->dev_t.pr_sum, size * sizeof(float));
		if (err > 0) {
			ERROR("CUDA MEMORY MALLOC ERROR!");
		}
		err = cudaMalloc((void**)&next_l->dev_t.po_sum, size * sizeof(float));
		if (err > 0) {
			ERROR("CUDA MEMORY MALLOC ERROR!");
		}

		next_l = next_l->follow;
	}
}

float* out_b = nullptr;
float* out_a = nullptr;
unsigned long code = 0;

void external(executable** head, executable** tail, unsigned long long gen) {
	if ((gen - 1)  % 10 == 0) {
		if (gen > 1) {
			layer_t* out = pick_layer(0);

			if (out_b == nullptr) {
				out_b = (float*)malloc(sizeof(float)*out->size);
			}

			if (out_a == nullptr) {
				out_a = (float*)malloc(sizeof(float)*out->size);
			}

			cudaMemcpy(out_b, out->out_b.t + NEURON_NUM / NEGITIVE_NEURON_CUT * sizeof(float), out->size * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(out_a, out->out_a.t + NEURON_NUM / NEGITIVE_NEURON_CUT * sizeof(float), out->size * sizeof(float), cudaMemcpyDeviceToHost);

			/*TO FIX:*/
			int i = 0;
			for (i = 0; i < out->size; i++) {
				out_b[i] *= out_b[i];
				if (out_b[i] / out_a[i] > 0.111) {
					if (out_a[i] > 0.555) {
						code += (1 << i);
					}
				}
				else {
					code = float(rand()) / float(RAND_MAX) * 4.0;
#ifdef DEBUG_EXTERNAL
					fprintf(stdout, "BEHAVIOR: RANDOMIZED.\n");
#endif
					break;
				}
			}
#ifdef DEBUG_EXTERNAL
			if (i == out->size) {
				fprintf(stdout, "BEHAVIOR: DETERMINED.\n");
			}
#endif

			friedrich_says(net_events(code), (char*)&code, sizeof(unsigned long));
		}

		friedrich_hearing();

		layer_t* in = pick_layer(0);
		prepend_executable(head, tail, new_executable(gen, EXECUTE_LAYER, in, in->next, in->next->t_layer));
	}
}

bool do_mutate(const unsigned long long gen) {
	return gen % 50 == 0;
}