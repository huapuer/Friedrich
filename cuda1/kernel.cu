/*
TODO: XOR时延特性
TODO: 打印DEBUG信息
TODO: 激发函数，初始值
TODO: HEBB权重增强

DESG: 加入逻辑层，逻辑层引用物理层的一部分或全部数据
	  物理层持有引用自己的逻辑层的引用，物理层被更新时负责同步更新逻辑层状态，并检查逻辑层输出并添加任务到调度器
DESG: Host Scheduler与Slave Batch进行解耦，实现Host与Slave并行作业，隐藏Host端调度开销
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "mpi.h" 
#include <Windows.h>
#include <memory.h>

#define error(stream,format,...) do{fprintf(stream,format,##__VA_ARGS__);system("pause");exit(1);}while(0)
#define DEBUG

enum layer_type {
	LAYER_I,
	LAYER_C,
	LAYER_XOR,
	LAYER_V
};

enum execute_type {
	EXECUTE_LAYER,
	EXECUTE_LINK,
	EXECUTE_INTEGRATE
};

struct gen_t {
	long long gen;
	float t;
};

struct gen_w {
	long long gen;
	long long working_gen;
	int stage;
	float t;
};

struct link;

struct layer_t {
	long long gen;
	int id;
	layer_type type;
	int size;
	int integrating_batch;
	int working_batch;
	gen_t *t;
	gen_t *dev_t;
	const float* dev_atte;
	link* pre;
	link* next;
	layer_t* follow;
};

struct link {
	long long gen;
	layer_t* layer;
	int size;
	gen_w* t;
	gen_w *dev_t;
	link* another;
};

struct executable {
	long long gen;
	execute_type type;
	layer_t* s;
	link* l;
	layer_t* t;
	executable* pre;
	executable* next;
};

struct map {
	int size;
	float* dev_t;
};

__constant__ map w_mutes;
int thread_num;
layer_t* list = 0;
layer_t* head = 0;

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void inte_i(gen_t* s) {
	int i = threadIdx.x;
	if (s[i].t > 3.0) {
		s[i].t = 1.0;
	}else {
		s[i].t = 0.0;
	}
}

__global__ void inte_c(gen_t* s) {
	int i = threadIdx.x;
	if (s[i].t > 3.0) {
		s[i].t = 1.0;
	}else {
		s[i].t = 0.0;
	}
}

__global__ void inte_xor(gen_t* s) {
	int i = threadIdx.x;
	if (s[i].t == 2.0) {
		s[i].t = 1.0;
	}else {
		s[i].t = 0.0;
	}
}

__global__ void mute_w(gen_w* w, const long long gen) {
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
}

__global__ void i_2_atte_i(const gen_t *s, gen_t *t, gen_w* w, const int ss, const long long gen)
{
	int i = threadIdx.x;
	for (int j = 0; j < ss; j++) {
		t[i].t = 0.0;
		if (s[j].t > 0.0) {
			t[i].t += s[j].t * w[i*j + j].t;
			w[i*j + j].working_gen = gen;
		}
	}
}

__global__ void i_2_i(const gen_t *s, gen_t *t, gen_w* w, const int ss, const long long gen)
{
	int i = threadIdx.x;
	for (int j = 0; j < ss; j++) {
		if (s[j].t > 0.0) {
			t[i].t += s[j].t * w[i*j + j].t;
			w[i*j + j].working_gen = gen;
		}
	}
}

__global__ void i_2_c(const gen_t *s, gen_t *t, const float* atte, const long long gen)
{
	int i = threadIdx.x;
	int gap = gen - t[i].gen;
	if (gap > 10) {
		t[i].t = 0.0;
	}
	else {
		t[i].t = t[i].t * pow((double)*atte, (double)gap);
	}
	t[i].t += s[i].t;
}

__global__ void i_2_atte_xor(const gen_t *s, gen_t *t)
{
	int i = threadIdx.x;
	t[i].t = 0.0;
	t[i].t += s[i].t > 0 ? 1.0 : 0.0;
}

__global__ void i_2_xor(const gen_t *s, gen_t *t)
{
	int i = threadIdx.x;
	t[i].t += s[i].t > 0 ? 1.0 : 0.0;
}

__global__ void c_2_atte_i(const gen_t *s, gen_t *t)
{
	int i = threadIdx.x;
	t[i].t = 0.0;
	t[i].t -= s[i].t;
}

__global__ void c_2_i(const gen_t *s, gen_t *t)
{
	int i = threadIdx.x;
	t[i].t -= s[i].t;
}

__global__ void xor_2_atte_i(const gen_t *s, gen_t *t)
{
	int i = threadIdx.x;
	t[i].t = 0.0;
	t[i].t += s[i].t;
}

__global__ void xor_2_i(const gen_t *s, gen_t *t)
{
	int i = threadIdx.x;
	t[i].t += s[i].t;
}

__global__ void v_2_atte_xor(const gen_t *s, gen_t *t)
{
	int i = threadIdx.x;
	t[i].t = 0.0;
	t[i].t += s[i].t > 0 ? 1.0 : 0.0;
	//printf("DEBUG v_2_atte_xor %d: f=%f\n", threadIdx.x, t[i].t);
}

__global__ void v_2_xor(const gen_t *s, gen_t *t)
{
	int i = threadIdx.x;
	t[i].t += s[i].t > 0 ? 1.0 : 0.0;
}

layer_t* new_layer(int id, layer_type type, int size,float atte=0.0) {
	layer_t* ret = (layer_t*)malloc(sizeof(layer_t));
	memset(ret, 0, sizeof(layer_t));
	ret->id = id;
	ret->type = type;
	ret->size = size;
	ret->gen = 0;
	ret->working_batch = 0;
	if (atte > 0.0) {
		cudaMemcpyToSymbol(ret->dev_atte, &atte, sizeof(int));
	}
	if (size > 0) {
		ret->t = (gen_t*)malloc(sizeof(gen_t)*size);
		//TODO: initialize gen_t?
		cudaMalloc((void**)&ret->dev_t, size * sizeof(gen_t));
		cudaMemcpy(ret->dev_t, ret->t, size * sizeof(gen_t), cudaMemcpyHostToDevice);
	}
	if (!list) {
		list = ret;
	}
	else {
		layer_t* iter = list;
		while (iter->follow) {
			iter = iter->follow;
		}
		iter->follow = ret;
	}
	return ret;
}

link* new_link(layer_t* layer, int size) {
	link* ret = (link*)malloc(sizeof(link));
	memset(ret, 0, sizeof(link));
	ret->layer = layer;
	ret->size = size;
	ret->t = (gen_w*)malloc(sizeof(gen_w)*size);
	//TODO: initialize gen_t
	cudaMalloc((void**)&ret->dev_t, size * sizeof(gen_w));
	cudaMemcpy(ret->dev_t, ret->t, size * sizeof(gen_w), cudaMemcpyHostToDevice);
	return ret;
}

layer_t* pick_layer(int idx) {
	if (!list) {
		error(stderr, "COMPILE ERROR: LAYER[%d] NOT EXSISTS!\n", idx);
	}
	else {
		if (list->id == idx) {
			return list;
		}
		else {
			layer_t* iter = list;
			while (iter->follow) {
				if (iter->follow->id == idx) {
					return iter->follow;
				}
				iter = iter->follow;
			}
		}
	}
	error(stderr, "COMPILE ERROR: LAYER[%d] NOT EXSISTS!\n", idx);
}

void add_link(link** head, link* next) {
	if(!*head) {
		*head = next;
		return;
	}
	else {
		link* tail = *head;
		while (tail->another) {
			tail = tail->another;
		}
		tail->another = next;
	}
}

layer_t* has_t(layer_t* s, int or_another_s, layer_t* next, int or_another_next) {
	if (!head) {
		if (s) {
			head = s;
		}
		else {
			error(stderr, "COMPILE ERROR: HEAD OF LAYER NOT EXSISTS!\n");
		}
	}
	if (or_another_s) {
		s = pick_layer(or_another_s);
	}
	if (!s) {
		error(stderr, "COMPILE ERROR: LAYER[%d] NOT EXSISTS!\n", or_another_s);
	}
	if (or_another_next) {
		next = pick_layer(or_another_next);
	}
	if (!next) {
		error(stderr, "COMPILE ERROR: LAYER[%d] NOT EXSISTS!\n", or_another_next);
	}
	int size = 0;
	switch (s->type) {
	case LAYER_I:
		switch (next->type) {
		case LAYER_I:
			size = s->size*next->size;
			break;
		case LAYER_C:
			if (s->size != next->size) {
				error(stderr, "COMPILE ERROR: LAYER[%d] SIZE UNMATCHED!\n", next->id);
			}
			size = 0;
			break;
		case LAYER_XOR:
			if (next->pre && next->pre->another) {
				error(stderr, "COMPILE ERROR: XOR LAYER[%d] OUT OF INPUTS!\n", next->id);
			}
			size = 0;
			break;
		default:
			error(stderr, "COMPILE ERROR: LAYER[%d] TYPE UNMATCHED!\n", next->id);
		}
		break;
	case LAYER_C:
		switch (next->type) {
		case LAYER_I:
			if (s->size != next->size) {
				error(stderr, "COMPILE ERROR: LAYER[%d] SIZE UNMATCHED!\n", next->id);
			}
			size = 0;
			break;
		default:
			error(stderr, "COMPILE ERROR: LAYER[%d] TYPE UNMATCHED!\n", next->id);
		}
		break;
	case LAYER_XOR:
		switch (next->type) {
		case LAYER_I:
			size = s->size*next->size;
			break;
		default:
			error(stderr, "COMPILE ERROR: LAYER[%d] TYPE UNMATCHED!\n", next->id);
		}
		break;
	case LAYER_V:
		switch (next->type) {
		case LAYER_XOR:
			if (s->size != next->size) {
				error(stderr, "COMPILE ERROR: LAYER[%d] SIZE UNMATCHED!\n", next->id);
			}
			if (next->pre && next->pre->another) {
				error(stderr, "COMPILE ERROR: XOR LAYER[%d] OUT OF INPUTS!\n", next->id);
			}
			size = 0;
			break;
		default:
			error(stderr, "COMPILE ERROR: LAYER[%d] TYPE UNMATCHED!\n", next->id);
		}
		break;
	default:
		error(stderr, "COMPILE ERROR: LAYER_TYPE[%d] UNDEFINED!\n", s->type);
	}
	link* l = new_link(next, size);
	add_link(&s->next, l);
	//add_link(&next->pre, l);
	return next;
}

executable* new_executable(int gen, execute_type type, layer_t* s, link* l, layer_t* t) {
	executable* ret = (executable*)malloc(sizeof(executable));
	memset(ret, 0, sizeof(executable));
	ret->gen = gen;
	ret->type = type;
	ret->s = s;
	ret->l = l;
	ret->t = t;
	return ret;
}

int launch_job(executable** head, executable** tail, executable* e, int gen, cudaStream_t stream) {
	if (e->type == EXECUTE_LINK) {
		mute_w <<<e->l->size / thread_num + 1, e->l->size>thread_num?thread_num:e->l->size, 0, stream >>> (e->l->dev_t, gen);
		return 1;
	}
	switch (e->s->type) {
	case LAYER_I:
		if (e->s->gen < gen) {
			inte_i <<<e->s->size / thread_num + 1, e->s->size>thread_num?thread_num:e->s->size, 0, stream>>> (e->s->dev_t);
			e->s->gen = gen;
			return 0;
		}
		else {
			switch (e->t->type) {
			case LAYER_I:
				if (e->t->gen < gen) {
					i_2_atte_i <<<e->t->size / thread_num + 1, e->t->size>thread_num?thread_num:e->t->size, 0, stream >>> (e->s->dev_t, e->t->dev_t, e->l->dev_t, e->s->size, gen);
					//e->t->gen = gen;
				}
				else {
					i_2_i <<<e->t->size / thread_num + 1, e->t->size>thread_num?thread_num:e->t->size, 0, stream >>> (e->s->dev_t, e->t->dev_t, e->l->dev_t, e->s->size, gen);
				}
				if (e->l->gen < gen) {
					executable* n = new_executable(gen + 1, EXECUTE_LINK, e->s, e->l, e->l->layer);
					/*
					if (head) {
						head->pre = n;
						head->pre->next = head;
						head = head->pre;
					}
					else {
						head = n;
					}
					*/
					if (*tail) {
						(*tail)->next = n;
						(*tail)->next->pre = *tail;
						(*tail) = (*tail)->next;
					}
					else {
						(*tail) = n;
					}
					e->l->gen = gen;
				}
				break;
			case LAYER_C:
				i_2_c <<<e->t->size / thread_num + 1, e->t->size>thread_num?thread_num:e->t->size, 0, stream >>>(e->s->dev_t, e->t->dev_t, e->t->dev_atte, gen);
				break;
			case LAYER_XOR:
				if (e->t->gen < gen) {
					i_2_atte_xor <<<e->t->size / thread_num + 1, e->t->size>thread_num?thread_num:e->t->size, 0, stream >>> (e->s->dev_t, e->t->dev_t);
					//e->t->gen = gen;
				}
				else {
					i_2_xor <<<e->t->size / thread_num + 1, e->t->size>thread_num?thread_num:e->t->size, 0, stream >>> (e->s->dev_t, e->t->dev_t);
				}
				break;
			default:
				error(stderr, "EXECUTE ERROR: LAYER_TYPE[%d] UNDEFINED OR UNMATCHED!\n", e->t->type);
			}
			return 1;
		}
		break;
	case LAYER_C:
		if (e->s->gen < gen) {
			inte_c <<<e->s->size / thread_num + 1, e->s->size>thread_num?thread_num:e->s->size, 0, stream >>> (e->s->dev_t);
			e->s->gen = gen;
			return 0;
		}
		else {
			switch (e->t->type) {
			case LAYER_I:
				if (e->t->gen < gen) {
					c_2_atte_i <<<e->t->size / thread_num + 1, e->t->size>thread_num?thread_num:e->t->size, 0, stream >>> (e->s->dev_t, e->t->dev_t);
					//e->t->gen = gen;
				}
				else {
					c_2_i <<<e->t->size / thread_num + 1, e->t->size>thread_num?thread_num:e->t->size, 0, stream >>> (e->s->dev_t, e->t->dev_t);
				}
				break;
			default:
				error(stderr, "EXECUTE ERROR: LAYER_TYPE[%d] UNDEFINED OR UNMATCHED!\n", e->t->type);
			}
			return 1;
		}
		break;
	case LAYER_XOR:
		if (e->s->gen < gen) {
			inte_xor <<<e->s->size / thread_num + 1, e->s->size>thread_num?thread_num:e->s->size, 0, stream >>> (e->s->dev_t);
			e->s->gen = gen;
			return 0;
		}
		else {
			switch (e->t->type) {
			case LAYER_I:
				if (e->t->gen < gen) {
					xor_2_atte_i <<<e->t->size / thread_num + 1, e->t->size>thread_num?thread_num:e->t->size, 0, stream >>> (e->s->dev_t, e->t->dev_t);
					//e->t->gen = gen;
				}
				else {
					xor_2_i <<<e->t->size / thread_num + 1, e->t->size>thread_num?thread_num:e->t->size, 0, stream >>> (e->s->dev_t, e->t->dev_t);
				}
				break;
			default:
				error(stderr, "EXECUTE ERROR: LAYER_TYPE[%d] UNDEFINED OR UNMATCHED!\n", e->t->type);
			}
			return 1;
		}
		break;
	case LAYER_V:
		switch (e->t->type) {
		case LAYER_XOR:
			if (e->t->gen < gen) {
				v_2_atte_xor <<<e->t->size / thread_num + 1, e->t->size>thread_num?thread_num:e->t->size, 0, stream >>> (e->s->dev_t, e->t->dev_t);
				//e->t->gen = gen;
			}
			else {
				v_2_xor <<<e->t->size / thread_num + 1, e->t->size>thread_num?thread_num:e->t->size, 0, stream >>> (e->s->dev_t, e->t->dev_t);
			}
			return 1;
		default:
			error(stderr, "EXECUTE ERROR: LAYER_TYPE[%d] UNDEFINED OR UNMATCHED!\n", e->t->type);
		}
		break;
	default:
		error(stderr, "EXECUTE ERROR: LAYER_TYPE[%d] UNDEFINED!\n", e->s->type);
	}
}

void execute(executable* e, int max_gen) {
	long long gen = 1;
	long long batch = 0;
	const int task_width = 10;
	cudaStream_t streams[task_width];
	for (int i = 0; i<task_width; i++)
	{
		cudaStreamCreate(&streams[i]);
	}
	executable* head = e;
	executable* tail = e;
	while (head) {
		//critical region begin
		//excute
		if (head->gen > gen) {
#ifdef DEBUG
			if (gen >= max_gen) {
				break;
			}
#endif
			gen++;
		}
		batch++;
		int tasks = 0;
		e = head;
		while (e && tasks < task_width) {
			if (e->gen == gen && tasks < task_width) {
				if (e->type == EXECUTE_LINK || (e->type==EXECUTE_LAYER && e->s->integrating_batch < batch && e->t->working_batch < batch)) {
					if (launch_job(&head, &tail, e, gen, streams[tasks]) == 1) {
#ifdef DEBUG
						fprintf(stdout, "GEN:%d BATCH:%d JOB:%d FROM:%d TO:%d FROM_WORKIN:%d TO_WORKING:%d\n", gen, batch, e->type, e->s->id, e->t->id, e->s->working_batch, e->t->working_batch);
#endif
						e->t->working_batch = batch;
						if (e->pre) {
							e->pre->next = e->next;
							if (!e->pre->pre) {
								head = e->pre;
								if (head) {
									head->next = e->next;
								}
							}
						}
						else {
							head = e->next;
							if (head) {
								head->next = e->next->next;
							}
						}
						if (e->next) {
							e->next->pre = e->pre;
							if (!e->next->next) {
								tail = e->next;
								if (tail) {
									tail->pre = e->pre;
								}
							}
						}
						else {
							tail = e->pre;
							if (tail) {
								tail->pre = e->pre->pre;
							}
						}
						if (e->type == EXECUTE_LAYER && e->t->gen < gen) {
							link* next = e->t->next;
							while (next) {
								executable* n = new_executable(gen + 1, EXECUTE_LAYER, e->t, next, next->layer);
								if (!head) {
									head = n;
								}
								if (tail) {
									tail->next = n;
									tail->next->pre = tail;
									tail = tail->next;
								}
								else {
									tail = n;
								}
								next = next->another;
							}
							e->t->gen = gen;
						}
						executable* f = e;
						e = e->next;
						free(f);
						continue;
					}
					else {
						e->s->integrating_batch = batch;
#ifdef DEBUG
						fprintf(stdout, "GEN:%d BATCH:%d JOB:%d FROM:%d TO:%d\n", gen, batch, EXECUTE_INTEGRATE, e->s->id, e->t->id);
#endif
					}
					tasks++;
				}
			}
			else {
				break;
			}
			e = e->next;
		}
		for (int i = 0; i < tasks; i++) {
			cudaStreamSynchronize(streams[i]);
		}
		//critical region end
	}
	for (int i = 0; i<5; i++)
	{
		cudaStreamDestroy(streams[i]);
	}
}

void emmit_layer(layer_t* l, float* t) {
	cudaMemcpy(l->dev_t, t, l->size * sizeof(gen_t), cudaMemcpyHostToDevice);
}

int main(int argc, char* argv[])
{
	/*
	int myid, numproces;
	int namelen;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numproces);
	MPI_Get_processor_name(processor_name, &namelen);
	char rbuf[10];
	if (myid == 4) {
		MPI_Recv(rbuf, 0, MPI_CHAR, 1, 0, MPI_COMM_WORLD, new MPI_Status());
		printf(rbuf);
	}
	if (myid == 1) {
		MPI_Send("hello send", 0, MPI_CHAR, 4, 0, MPI_COMM_WORLD);
	}
	Sleep(10000);
	MPI_Barrier(MPI_COMM_WORLD);
	*/

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		error(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, 0);
	thread_num = properties.maxThreadsPerBlock;

	w_mutes.size = 10;
	float host_t[10] = { 1.0 };
	cudaMemcpyToSymbol(w_mutes.dev_t, &host_t, w_mutes.size * sizeof(gen_w));

	has_t(new_layer(0, LAYER_V, 1), 0, new_layer(1, LAYER_XOR, 1), 0);
	has_t(NULL, 1, new_layer(2, LAYER_I, 2), 0);
	has_t(NULL, 2, new_layer(3, LAYER_I, 2), 0);
	has_t(NULL, 3, new_layer(4, LAYER_I, 2), 0);
	has_t(NULL, 4, new_layer(5, LAYER_I, 1), 0);
	has_t(NULL, 5, NULL, 1);
	has_t(NULL, 4, new_layer(6, LAYER_I, 2), 0);
	has_t(NULL, 3, new_layer(7, LAYER_C, 2), 0);
	has_t(NULL, 7, NULL, 4);

	float input = 1.0;
	emmit_layer(head, &input);

	execute(new_executable(1, EXECUTE_LAYER, head, head->next, head->next->layer),20);

	// Copy output vector from GPU buffer to host memory.
	//cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	//if (cudaStatus != cudaSuccess) {
	//	error(stderr, "cudaMemcpy failed!");
	//	goto Error;
	//}

	//printf("hello world! process %d of %d on %s\n", myid, numproces, processor_name);
	//MPI_Finalize();
	//system("pause");

Error:
	/*
	layer_t* iter = head;
	while (iter) {
		cudaFree(iter->dev_t);
		link* iter2 = iter->next;
		while (iter2) {
			cudaFree(iter2->dev_t);
			iter2 = iter2->another;
		}
		iter2 = iter->pre;
		while (iter2) {
			cudaFree(iter2->dev_t);
			iter2 = iter2->another;
		}
	}
	*/
	system("pause");
    return 0;
}
