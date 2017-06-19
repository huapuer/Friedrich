/*
TODO: ������������ʼֵ
TODO: HEBBȨ����ǿ

DESG: �����߼��㣬�߼�������������һ���ֻ�ȫ������
	  �������������Լ����߼�������ã�����㱻����ʱ����ͬ�������߼���״̬��������߼��������������񵽵�����
DESG: Host Scheduler��Slave Batch���н��ʵ��Host��Slave������ҵ������Host�˵��ȿ���

TODO: ����������������߼���֮���״̬ͬ���߼�(scheduling debug), ������ɣ��������߼�������������(layer1->layer2)
TODO: ���Ӳ�ͬ���ӷ�ʽ��1:1/n:n��
TODO: ���Ӳ�����Ȩ������֧�֣�mute_fnΪNULL��
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
	EXECUTE_LINK
};

struct gen_t {
	unsigned long long gen;
	float t;
};

struct gen_w {
	unsigned long long gen;
	unsigned long long working_gen;
	int stage;
	float t;
};

struct link;

typedef void(*fp_integrate)(gen_t*, int);
typedef void(*fp_mute)(gen_w*, const unsigned long long);
typedef void(*fp_clear_push)(const gen_t *, gen_t *, int, gen_w*, const int, const unsigned long long);
typedef void(*fp_push)(const gen_t*, gen_t*, int, gen_w*, const int, const unsigned long long);

struct layer_t {
	int id;
	layer_type type;
	int size;
	link* pre;
	link* next;
	layer_t* follow;
	fp_integrate integrate_fn;
	fp_clear_push clear_push_fn;
	fp_push push_fn;

	//phsical
	unsigned long long gen;
	unsigned long long working_gen;
	unsigned long long integrated_gen;
	unsigned long long swap_gen;
	unsigned long long working_batch;
	unsigned long long integrating_batch;
	gen_t *t;
	int cur_s_dev_t;
	int cur_t_dev_t;
	gen_t *dev_t[2];
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
	unsigned long long gen;
	unsigned long long mutated_gen;
	unsigned long long mutating_batch;
	layer_t* layer;
	int size;
	gen_w* t;
	gen_w *dev_t;
	link* another;
};

struct executable {
	unsigned long long gen;
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

__global__ void default_mute(gen_w* w, const unsigned long long gen) {
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

__global__ void default_clear_push(const gen_t *s, gen_t *t, int to, gen_w* w, const int ss, const unsigned long long gen)
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

__global__ void default_push(const gen_t *s, gen_t *t, int to, gen_w* w, const int ss, const unsigned long long gen)
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
	ret->working_gen = 0;
	ret->integrated_gen = 0;
	ret->swap_gen = 0;
	ret->working_batch = 0;
	ret->integrating_batch = 0;
	ret->offset = 0;
	ret->integrate_fn = inte_fn;
	ret->clear_push_fn = cl_p_fn;
	ret->push_fn = p_fn;
	ret->cur_s_dev_t = 0;
	ret->cur_t_dev_t = 1;
	ret->phsical = ret;

	if (atte > 0.0) {
		cudaMemcpyToSymbol(ret->dev_atte, &atte, sizeof(int));
	}
	if (size > 0) {
		ret->t = (gen_t*)malloc(sizeof(gen_t)*size);
		//TODO: initialize gen_t?
		cudaMalloc((void**)&ret->dev_t[0], size * sizeof(gen_t));
		cudaMemcpy(ret->dev_t[0], ret->t, size * sizeof(gen_t), cudaMemcpyHostToDevice);
		cudaMalloc((void**)&ret->dev_t[1], size * sizeof(gen_t));
		cudaMemcpy(ret->dev_t[1], ret->t, size * sizeof(gen_t), cudaMemcpyHostToDevice);
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
	ret->delegate = delegate;

	layer_t* pl= pick_layer(phsical);
	ret->phsical = pl;
	ret->offset = offset;
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
	ret->mutated_gen = 0;
	ret->mutating_batch = 0;
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

void append_executable(executable** head, executable** tail, executable* n) {
	if (!*head) {
		*head = n;
	}
	if (*tail) {
		(*tail)->next = n;
		(*tail)->next->pre = *tail;
		(*tail) = (*tail)->next;
	}
	else {
		*tail = n;
	}
}

void remove_executable(executable** head, executable** tail, executable* e) {
	if (e->pre) {
		e->pre->next = e->next;
		if (!e->pre->pre) {
			*head = e->pre;
			if (*head) {
				(*head)->next = e->next;
			}
		}
	}
	else {
		*head = e->next;
		if (*head) {
			(*head)->next = e->next->next;
		}
	}
	if (e->next) {
		e->next->pre = e->pre;
		if (!e->next->next) {
			*tail = e->next;
			if (*tail) {
				(*tail)->pre = e->pre;
			}
		}
	}
	else {
		*tail = e->pre;
		if (*tail) {
			(*tail)->pre = e->pre->pre;
		}
	}
}

void swap_layer_dev(layer_t* l) {
	int tmp;
	tmp = l->cur_s_dev_t;
	l->cur_s_dev_t = l->cur_t_dev_t;
	l->cur_t_dev_t = tmp;
}

void wrap_layers(executable* task, layer_t** s_phisical, layer_t** t_phisical, layer_t** s_logical, layer_t** t_logical) {
	switch (task->s->type) {
	case LAYER_PHSICAL:
		*s_phisical = task->s;
		*s_logical = *s_phisical;
		break;
	case LAYER_LOGICAL:
		*s_phisical = task->s->phsical;
		switch (task->s->delegate) {
		case true:
			*s_logical = *s_phisical;
			break;
		case false:
			*s_logical = task->s;
			break;
		}
		break;
	}
	switch (task->t->type) {
	case LAYER_PHSICAL:
		*t_phisical = task->t;
		*t_logical = *t_phisical;
		break;
	case LAYER_LOGICAL:
		*t_phisical = task->t->phsical;
		switch (task->t->delegate) {
		case true:
			*t_logical = *t_phisical;
			break;
		case false:
			*t_logical = task->t;
			break;
		}
		break;
	}

}

void execute(executable* head, int max_gen) {
	unsigned long long gen = 1;
	unsigned long long batch = 0;
	const int task_width = 10;
	int tasks = 0;
	cudaStream_t streams[task_width];
	for (int i = 0; i<task_width; i++)
	{
		cudaStreamCreate(&streams[i]);
	}

	executable* link_task_head = NULL;
	executable* link_task_tail = NULL;
	executable* layer_task_head = head;
	executable* layer_task_tail = head;

	while (layer_task_head) {
		//critical region begin
		//excute
		
		batch++;
		tasks = 0;

		if (layer_task_head->gen > gen) {
#ifdef DEBUG
			if (gen >= max_gen) {
				break;
			}
#endif
			gen++;

			executable* link_task = link_task_head;
			while (link_task) {			
				while (link_task && tasks < task_width) {
					mute_fn << <link_task->l->size / thread_num + 1, link_task->l->size>thread_num ? thread_num : link_task->l->size, 0, streams[tasks] >> >(link_task->l->dev_t, gen);
					remove_executable(&link_task_head, &link_task_tail, link_task);
					tasks++;
					link_task->l->mutating_batch = batch;
#ifdef DEBUG
					layer_t *s_phisical, *t_phisical, *s_logical, *t_logical;
					wrap_layers(link_task, &s_phisical, &t_phisical, &s_logical, &t_logical);
					fprintf(stdout, "GEN:%d BATCH:%d JOB:%s FROM:%d TO:%d\n", gen, batch, "MUTW", s_logical->id, t_logical->id);
#endif
					link_task = link_task->next;
				}
				if (tasks == task_width) {
					for (int i = 0; i < tasks; i++) {
						cudaStreamSynchronize(streams[i]);
					}
					batch++;
					tasks = 0;
				}
			}

			pick_layer(0)->integrated_gen = gen;
			layer_task_head->pre=new_executable(gen, EXECUTE_LAYER, pick_layer(0), pick_layer(0)->next, pick_layer(0)->next->layer);
			layer_task_head->pre->next = layer_task_head;
			layer_task_head = layer_task_head->pre;
		}

		executable* layer_task = layer_task_head;
		while (layer_task && tasks < task_width) {
			if (layer_task->gen == gen && tasks < task_width) {
				layer_t *s_phisical, *t_phisical, *s_logical, *t_logical;
				wrap_layers(layer_task, &s_phisical, &t_phisical, &s_logical, &t_logical);
				if (s_phisical->swap_gen != gen) {
					swap_layer_dev(s_phisical);
					s_phisical->swap_gen = gen;
				}
				if (t_phisical->swap_gen != gen) {
					swap_layer_dev(t_phisical);
					t_phisical->swap_gen = gen;
				}
				if (s_logical->integrated_gen != gen) {
					if (s_phisical->integrated_gen != gen) {
						layer_task->s->integrate_fn << <s_logical->size / thread_num + 1, s_logical->size>thread_num ? thread_num : s_logical->size, 0, streams[tasks] >> > (s_phisical->dev_t[s_phisical->cur_s_dev_t], s_logical->offset);
						tasks++;
						s_phisical->integrated_gen = gen;
					}
					s_logical->integrated_gen = gen;
					s_logical->integrating_batch = batch;
#ifdef DEBUG
					fprintf(stdout, "GEN:%d BATCH:%d JOB:%s FROM:%d TO:%d BUFF:%d\n", gen, batch, "INTE", s_logical->id, t_logical->id, s_phisical->cur_s_dev_t);
#endif
				}
				else if(s_logical->integrating_batch != batch){
					if (layer_task->l->mutating_batch != batch) {
						if (t_logical->working_gen != gen) {
							layer_task->s->clear_push_fn << <t_logical->size / thread_num + 1, t_logical->size>thread_num ? thread_num : t_logical->size, 0, streams[tasks] >> > (s_phisical->dev_t[s_phisical->cur_s_dev_t], t_phisical->dev_t[t_phisical->cur_t_dev_t], t_logical ->offset, layer_task->l->dev_t, s_logical->size, gen);
							tasks++;
							remove_executable(&layer_task_head, &layer_task_tail, layer_task);
							link* next_link = t_logical->next;
							while (next_link) {
								executable* n = new_executable(gen + 1, EXECUTE_LAYER, t_logical, next_link, next_link->layer);
								append_executable(&layer_task_head, &layer_task_tail, n);
								next_link = next_link->another;
							}
							if (t_logical != t_phisical) {
								if (t_phisical->working_gen != gen) {
									link* next_link = t_phisical->next;
									while (next_link) {
										executable* n = new_executable(gen + 1, EXECUTE_LAYER, t_phisical, next_link, next_link->layer);
										append_executable(&layer_task_head, &layer_task_tail, n);
										next_link = next_link->another;
									}
								}
							}
							else {
								layer_t* next = t_logical->logical_head;
								while (next) {
									if (!next->delegate) {
										link* next_link = next->next;
										while (next_link) {
											executable* n = new_executable(gen + 1, EXECUTE_LAYER, next, next_link, next_link->layer);
											append_executable(&layer_task_head, &layer_task_tail, n);
											next_link = next_link->another;
										}
									}
									next->working_gen = gen;
									next->working_batch = batch;
									next = next->next_logical;
								}
							}
							t_logical->working_gen = gen;
							t_logical->working_batch = batch;
							t_phisical->working_gen = gen;
							t_phisical->working_batch = batch;
#ifdef DEBUG
							fprintf(stdout, "GEN:%d BATCH:%d JOB:%s FROM:%d TO:%d BUFF:%d\n", gen, batch, "CLRP", s_logical->id, t_logical->id, t_phisical->cur_t_dev_t);
#endif
						}
						else if (t_logical->working_batch != batch){
							layer_task->s->push_fn << <t_logical->size / thread_num + 1, t_logical->size>thread_num ? thread_num : t_logical->size, 0, streams[tasks] >> > (s_phisical->dev_t[s_phisical->cur_s_dev_t], t_phisical->dev_t[t_phisical->cur_t_dev_t], t_logical->offset, layer_task->l->dev_t, s_logical->size, gen);
							tasks++;
							remove_executable(&layer_task_head, &layer_task_tail, layer_task);
							t_logical->working_batch = batch;
#ifdef DEBUG
							fprintf(stdout, "GEN:%d BATCH:%d JOB:%s FROM:%d TO:%d BUFF:%d\n", gen, batch, "PUSH", s_logical->id, t_logical->id, t_phisical->cur_t_dev_t);
#endif
						}
						if (layer_task->l->mutated_gen != gen) {
							executable* n = new_executable(gen + 1, EXECUTE_LINK, s_logical, layer_task->l, layer_task->l->layer);
							append_executable(&link_task_head, &link_task_tail, n);
							layer_task->l->mutated_gen = gen;
						}
					}
				}
			}
			else {
				break;
			}
			layer_task = layer_task->next;
		}
		for (int i = 0; i < tasks; i++) {
			cudaStreamSynchronize(streams[i]);
		}
		//critical region end
	}
}

void emit_layer(layer_t* l, float* t) {
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
	//has_t(NULL, 210, NULL, 31);
	has_t(NULL, 22, NULL, 32);
	//has_t(NULL, 220, NULL, 32);
	has_t(NULL, 23, NULL, 33);
	//has_t(NULL, 230, NULL, 33);

	has_t(NULL, 21, NULL, 210);
	//has_t(NULL, 210, NULL, 21);
	has_t(NULL, 22, NULL, 220);
	//has_t(NULL, 220, NULL, 22);
	has_t(NULL, 23, NULL, 230);
	//has_t(NULL, 230, NULL, 23);

	has_t(NULL, 3, NULL, 30);
	//has_t(NULL, 30, NULL, 3);

	float input = 1.0;
	emit_layer(pick_layer(0), &input);

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
