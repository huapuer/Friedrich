/*
TODO: 激发函数，初始值
TODO: HEBB权重增强

DESG: 加入逻辑层，逻辑层引用物理层的一部分或全部数据
	  物理层持有引用自己的逻辑层的引用，物理层被更新时负责同步更新逻辑层状态，并检查逻辑层输出并添加任务到调度器
DESG: Host Scheduler与Slave Batch进行解耦，实现Host与Slave并行作业，隐藏Host端调度开销

TODO: 增加物理层与所属逻辑层之间的状态同步逻辑(scheduling debug), 部分完成，增加子逻辑层输出情况测试(layer1->layer2)
TODO: 增加不同连接方式（1:1/n:n）
TODO: 增加不更新权重连接支持（mute_fn为NULL）
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "mpi.h" 
#include <Windows.h>
#include <memory.h>

#define ERROR(format,...) do{fprintf(stderr,format,##__VA_ARGS__);system("pause");exit(1);}while(0)
#define DEBUG

enum layer_type {
	LAYER_PHSICAL,
	LAYER_LOGICAL
};

enum execute_type {
	EXECUTE_LAYER,
	EXECUTE_LINK,
	EXECUTE_INTEGRATE
};

enum launch_type {
	LAUNCH_INTEGRATE,
	LAUNCH_PUSH,
	LAUNCH_CLEAR_PUSH,
	LAUNCH_MUTE_W
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

typedef void(*fp_integrate)(gen_t*, int);
typedef void(*fp_mute)(gen_w*, const long long);
typedef void(*fp_clear_push)(const gen_t *, gen_t *, int, gen_w*, const int, const long long);
typedef void(*fp_push)(const gen_t*, gen_t*, int, gen_w*, const int, const long long);

struct layer_t {
	long long gen;
	int id;
	layer_type type;
	int size;
	int integrating_batch;
	int working_batch;
	link* pre;
	link* next;
	layer_t* follow;
	fp_integrate integrate_fn;
	fp_clear_push clear_push_fn;
	fp_push push_fn;

	//phsical
	gen_t *t;
	gen_t *dev_t;
	const float* dev_atte;
	layer_t* logical_head;
	layer_t* logical_tail;

	//logical
	int offset;
	bool delegate;
	layer_t* phsical;
	layer_t* next_logical;
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
layer_t* layer_list = 0;

__global__ void default_integrate(gen_t* s, int offset) {
	int i = threadIdx.x + offset;
	if (s[i].t > 3.0) {
		s[i].t = 1.0;
	}else {
		s[i].t = 0.0;
	}
}

__global__ void default_mute(gen_w* w, const long long gen) {
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

__global__ void default_clear_push(const gen_t *s, gen_t *t, int to, gen_w* w, const int ss, const long long gen)
{
	int i = threadIdx.x + to;
	for (int j = 0; j < ss; j++) {
		t[i].t = 0.0;
		if (s[j].t > 0.0) {
			t[i].t += s[j].t * w[i*j + j].t;
			w[i*j + j].working_gen = gen;
		}
	}
}

__global__ void default_push(const gen_t *s, gen_t *t, int to, gen_w* w, const int ss, const long long gen)
{
	int i = threadIdx.x + to;
	for (int j = 0; j < ss; j++) {
		if (s[j].t > 0.0) {
			t[i].t += s[j].t * w[i*j + j].t;
			w[i*j + j].working_gen = gen;
		}
	}
}

layer_t* pick_layer(int idx) {
	if (!layer_list) {
		ERROR("COMPILE ERROR: LAYER[%d] NOT EXSISTS!\n", idx);
	}
	else {
		if (layer_list->id == idx) {
			return layer_list;
		}
		else {
			layer_t* iter = layer_list;
			while (iter->follow) {
				if (iter->follow->id == idx) {
					return iter->follow;
				}
				iter = iter->follow;
			}
		}
	}
	ERROR("COMPILE ERROR: LAYER[%d] NOT EXSISTS!\n", idx);
}

layer_t* new_layer_phsical(int id, int size,float atte=0.0, fp_integrate inte_fn=default_integrate, fp_clear_push cl_p_fn=default_clear_push, fp_push p_fn=default_push) {
	layer_t* ret = (layer_t*)malloc(sizeof(layer_t));
	memset(ret, 0, sizeof(layer_t));
	ret->id = id;
	ret->type = LAYER_PHSICAL;
	ret->size = size;
	ret->gen = 0;
	ret->working_batch = 0;
	ret->offset = 0;
	ret->integrate_fn = inte_fn;
	ret->clear_push_fn = cl_p_fn;
	ret->push_fn = p_fn;

	if (atte > 0.0) {
		cudaMemcpyToSymbol(ret->dev_atte, &atte, sizeof(int));
	}
	if (size > 0) {
		ret->t = (gen_t*)malloc(sizeof(gen_t)*size);
		//TODO: initialize gen_t?
		cudaMalloc((void**)&ret->dev_t, size * sizeof(gen_t));
		cudaMemcpy(ret->dev_t, ret->t, size * sizeof(gen_t), cudaMemcpyHostToDevice);
	}
	if (!layer_list) {
		layer_list = ret;
	}
	else {
		layer_t* iter = layer_list;
		while (iter->follow) {
			iter = iter->follow;
		}
		iter->follow = ret;
	}
	return ret;
}

layer_t* new_layer_logical(int id, int phsical, int offset, int size, bool delegate) {
	layer_t* ret = (layer_t*)malloc(sizeof(layer_t));
	memset(ret, 0, sizeof(layer_t));
	ret->id = id;
	ret->type = LAYER_LOGICAL;
	ret->size = size;
	ret->gen = 0;
	ret->working_batch = 0;
	ret->delegate = delegate;

	layer_t* pl= pick_layer(phsical);
	ret->phsical = pl;
	ret->offset = offset;
	ret->t = pl->t;
	ret->dev_t = pl->dev_t;
	ret->integrate_fn = pl->integrate_fn;
	ret->clear_push_fn = pl->clear_push_fn;
	ret->push_fn = pl->push_fn;

	if (!pl->logical_head) {
		pl->logical_head = ret;
	}
	if (pl->logical_tail) {
		pl->logical_tail->next_logical = ret;
		pl->logical_tail = pl->logical_tail->next_logical;
	}
	else {
		pl->logical_tail = ret;
	}

	if (!layer_list) {
		layer_list = ret;
	}
	else {
		layer_t* iter = layer_list;
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
	if (!s) {
		s = pick_layer(or_another_s);
	}
	if (!s) {
		ERROR("COMPILE ERROR: LAYER[%d] NOT EXSISTS!\n", or_another_s);
	}

	if (!next) {
		next = pick_layer(or_another_next);
	}
	if (!next) {
		ERROR("COMPILE ERROR: LAYER[%d] NOT EXSISTS!\n", or_another_next);
	}
	int size = s->size*next->size;
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

fp_mute mute_fn;

int launch_job(executable** head, executable** tail, executable* e, int gen, cudaStream_t stream) {
	if (e->type == EXECUTE_LINK) {
		mute_fn <<<e->l->size / thread_num + 1, e->l->size>thread_num?thread_num:e->l->size, 0, stream >>> (e->l->dev_t, gen);
		return LAUNCH_MUTE_W;
	}
	if (e->s->gen < gen) {
		e->s->integrate_fn <<<e->s->size / thread_num + 1, e->s->size>thread_num?thread_num:e->s->size, 0, stream>>> (e->s->dev_t, e->s->offset);
		e->s->gen = gen;
		return LAUNCH_INTEGRATE;
	}
	else {
		int ret = LAUNCH_CLEAR_PUSH;
		if (e->t->gen < gen) {
			e->s->clear_push_fn <<<e->t->size / thread_num + 1, e->t->size>thread_num?thread_num:e->t->size, 0, stream >>> (e->s->dev_t, e->t->dev_t, e->t->offset, e->l->dev_t, e->s->size, gen);
			//e->t->gen = gen;
		}
		else {
			e->s->push_fn <<<e->t->size / thread_num + 1, e->t->size>thread_num?thread_num:e->t->size, 0, stream >>> (e->s->dev_t, e->t->dev_t, e->t->offset, e->l->dev_t, e->s->size, gen);
			ret = LAUNCH_PUSH;
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
		return ret;
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

	char* LAUNCH_TYPE_DBG_STR[4] = { "INTE","PUSH","CLRP","MUTW" };

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
		
		/*
		executable* next_e = head;
		while (next_e) {
			if (next_e->gen == gen) {
				break;
			}
			next_e = next_e->next;
		}
		if (!next_e) {
#ifdef DEBUG
			if (gen >= max_gen) {
				break;
			}
#endif
			gen++;
		}
		*/
		batch++;
		int tasks = 0;
		e = head;
		while (e && tasks < task_width) {
			if (e->gen == gen && tasks < task_width) {
				if (e->type == EXECUTE_LINK || (e->type==EXECUTE_LAYER && e->s->integrating_batch < batch && e->t->working_batch < batch)) {
					int ret = launch_job(&head, &tail, e, gen, streams[tasks]);
					if ( ret != LAUNCH_INTEGRATE) {
#ifdef DEBUG
						fprintf(stdout, "GEN:%d BATCH:%d JOB:%s FROM:%d TO:%d FROM_WORKIN:%d TO_WORKING:%d\n", gen, batch, LAUNCH_TYPE_DBG_STR[ret], e->s->id, e->t->id, e->s->working_batch, e->t->working_batch);
#endif
						e->t->working_batch = batch;
						switch (e->t->type) {
						case LAYER_LOGICAL:
							e->t->phsical->working_batch = batch;
							break;
						case LAYER_PHSICAL:
							layer_t* next = e->t->logical_head;
							while (next) {
								next->working_batch = batch;
								next = next->next_logical;
							}
							break;
						}
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
							switch (e->t->type) {
							case LAYER_LOGICAL: {
								if (e->t->phsical->gen < gen) {
									if (!e->t->delegate) {
										link* next = e->t->phsical->next;
										while (next) {
											executable* n = new_executable(gen + 1, EXECUTE_LAYER, e->t->phsical, next, next->layer);
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
									}
									e->t->phsical->gen = gen;
								}
								if (e->t->delegate) {
									layer_t* next = e->t->phsical->logical_head;
									while (next) {
										next->gen = gen;
										next = next->next_logical;
									}
								}
							}
								break;
							case LAYER_PHSICAL: {
								layer_t* next = e->t->logical_head;
								while (next) {
									if (next->gen < gen) {
										if (!next->delegate) {
											link* next_link = next->next;
											while (next_link) {
												executable* n = new_executable(gen + 1, EXECUTE_LAYER, next, next_link, next_link->layer);
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
												next_link = next_link->another;
											}
										}
										next->gen = gen;
									}
									next = next->next_logical;
								}
							}
								break;
							}
						}
						executable* f = e;
						e = e->next;
						free(f);
						continue;
					}
					else {
						e->s->integrating_batch = batch;
						switch (e->s->type) {
						case LAYER_LOGICAL:
							e->s->phsical->integrating_batch = batch;
							break;
						case LAYER_PHSICAL:
							layer_t* next = e->s->logical_head;
							while (next) {
								next->integrating_batch = batch;
								next = next->next_logical;
							}
							break;
						}
#ifdef DEBUG
						fprintf(stdout, "GEN:%d BATCH:%d JOB:%s FROM:%d TO:%d\n", gen, batch, LAUNCH_TYPE_DBG_STR[ret], e->s->id, e->t->id);
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
		ERROR("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, 0);
	thread_num = properties.maxThreadsPerBlock;

	w_mutes.size = 10;
	float host_t[10] = { 1.0 };
	cudaMemcpyToSymbol(w_mutes.dev_t, &host_t, w_mutes.size * sizeof(gen_w));

	mute_fn = default_mute;

	new_layer_phsical(0, 9);

	new_layer_phsical(1, 9);
	new_layer_logical(11, 1, 0, 3, false);
	new_layer_logical(12, 1, 3, 3, false);
	new_layer_logical(13, 1, 6, 3, false);

	new_layer_phsical(21, 3);
	new_layer_logical(210, 21, 0, 3, true);
	new_layer_phsical(22, 3);
	new_layer_logical(220, 22, 0, 3, true);
	new_layer_phsical(23, 3);
	new_layer_logical(230, 23, 0, 3, true);

	new_layer_phsical(3, 3);
	new_layer_logical(30, 3, 0, 3, true);
	new_layer_logical(31, 3, 0, 1, false);
	new_layer_logical(32, 3, 3, 1, false);
	new_layer_logical(33, 3, 6, 1, false);

	has_t(NULL, 0, NULL, 1);

	has_t(NULL, 11, NULL, 21);
	has_t(NULL, 12, NULL, 22);
	has_t(NULL, 13, NULL, 23);

	has_t(NULL, 21, NULL, 31);
	has_t(NULL, 210, NULL, 31);
	has_t(NULL, 22, NULL, 32);
	has_t(NULL, 220, NULL, 32);
	has_t(NULL, 23, NULL, 33);
	has_t(NULL, 230, NULL, 33);

	has_t(NULL, 21, NULL, 210);
	has_t(NULL, 210, NULL, 21);
	has_t(NULL, 22, NULL, 220);
	has_t(NULL, 220, NULL, 22);
	has_t(NULL, 23, NULL, 230);
	has_t(NULL, 230, NULL, 23);

	has_t(NULL, 3, NULL, 30);
	has_t(NULL, 30, NULL, 3);

	float input = 1.0;
	emmit_layer(pick_layer(1), &input);

	execute(new_executable(1, EXECUTE_LAYER, pick_layer(0), pick_layer(0)->next, pick_layer(0)->next->layer),200);

	// Copy output vector from GPU buffer to host memory.
	//cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	//if (cudaStatus != cudaSuccess) {
	//	ERROR("cudaMemcpy failed!");
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
