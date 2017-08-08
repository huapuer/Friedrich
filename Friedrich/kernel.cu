/*
TODO: 激发函数，初始值
TODO: HEBB权重增强

DESG: 加入逻辑层，逻辑层引用物理层的一部分或全部数据
	  物理层持有引用自己的逻辑层的引用，物理层被更新时负责同步更新逻辑层状态，并检查逻辑层输出并添加任务到调度器
DESG: Host Scheduler与Slave Batch进行解耦，实现Host与Slave并行作业，隐藏Host端调度开销

TODO: 增加物理层与所属逻辑层之间的状态同步逻辑(scheduling debug), 部分完成，增加子逻辑层输出情况测试(layer1->t_layer2)
TODO: 增加不同连接方式（1:1/n:n）
TODO: 增加不更新权重连接支持（mute_fn为NULL）
*/
#include <stdio.h>
#include "mpi.h" 
#include <Windows.h>
#include <memory.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../Ludwig/Ludwig/ludwig_net.h"
#pragma comment(lib, "../../Ludwig/x64/Debug/Ludwig.lib")
#include "executable.h"

#include "net.h"
#include "profile.h"

#define ERROR(format,...) do{fprintf(stderr,format,##__VA_ARGS__);system("pause");exit(1);}while(0)

typedef void(*fp_integrate)(gen_t*, int);
typedef void(*fp_mute)(gen_w*, const unsigned long long);
typedef void(*fp_clear_push)(const gen_t *, gen_t *, int, gen_w*, const int, const unsigned long long);
typedef void(*fp_push)(const gen_t*, gen_t*, int, gen_w*, const int, const unsigned long long);

int thread_num;
layer_t* layer_list = 0;

void append_updated_layer(layer_t** head, layer_t** tail, layer_t* n) {
	if (!*head) {
		*head = n;
	}
	if (*tail) {
		(*tail)->updated_next = n;
		(*tail)->updated_next->updated_pre = *tail;
		(*tail) = (*tail)->updated_next;
	}
	else {
		*tail = n;
	}
}

void remove_updated_layer(layer_t** head, layer_t** tail, layer_t* l) {
	if (l->updated_pre) {
		l->updated_pre->updated_next = l->updated_next;
		if (!l->updated_pre->updated_pre) {
			*head = l->updated_pre;
			if (*head) {
				(*head)->updated_next = l->updated_next;
			}
		}
	}
	else {
		*head = l->updated_next;
		if (*head) {
			(*head)->updated_next = l->updated_next->updated_next;
		}
	}
	if (l->updated_next) {
		l->updated_next->updated_pre = l->updated_pre;
		if (!l->updated_next->updated_next) {
			*tail = l->updated_next;
			if (*tail) {
				(*tail)->updated_pre = l->updated_pre;
			}
		}
	}
	else {
		*tail = l->updated_pre;
		if (*tail) {
			(*tail)->updated_pre = l->updated_pre->updated_pre;
		}
	}
	l->updated_next = NULL;
	l->updated_pre = NULL;
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

void execute(int max_gen) {
	unsigned long long gen = 1;
	unsigned long long batch = 0;
	const int task_width = 100;
	int tasks = 0;
	cudaStream_t streams[task_width];
	for (int i = 0; i<task_width; i++)
	{
		cudaStreamCreate(&streams[i]);
	}

	executable* layer_task_head = NULL;
	executable* layer_task_tail = NULL;
	layer_t* updated_layer_head = NULL;
	layer_t* updated_layer_tail = NULL;

	external(&layer_task_head, &layer_task_tail, gen);

	while (layer_task_head) {
		//critical region begin
		//excute
		
		batch++;
		tasks = 0;

		if (layer_task_head->gen > gen) {
#ifdef DEBUG_SCHEDULE
			if (gen >= max_gen) {
				break;
			}
#endif
			gen++;

			external(&layer_task_head, &layer_task_tail, gen);
		}

		executable* layer_task = layer_task_head;
		while (layer_task && tasks < task_width) {
			if (layer_task->gen == gen && tasks < task_width) {
				layer_t *s_phisical, *t_phisical, *s_logical, *t_logical;
				wrap_layers(layer_task, &s_phisical, &t_phisical, &s_logical, &t_logical);
				switch (layer_task->type) {
				case EXECUTE_JOINT:
					if (layer_task->s->integrating_batch != batch) {
						joint << <t_logical->size / thread_num + 1, t_logical->size>thread_num ? thread_num : t_logical->size, 0, streams[tasks] >> >(t_logical->dev_t[t_logical->cur_s_dev_t].t, t_logical->offset, layer_task->l->dev_t.r2, layer_task->l->dev_t.po);
#ifdef DEBUG_SCHEDULE
						fprintf(stdout, "GEN:%d BATCH:%d JOB:%s FROM:%d TO:%d BUFF:%d\n", gen, batch, "CUNT", s_logical->id, t_logical->id, s_logical->cur_s_dev_t);
#endif
						//tasks++;

						layer_task->l->counting_batch = batch;

						if (do_mutate(gen)) {
							executable* n = new_executable(gen, EXECUTE_LINK, s_logical, layer_task->l, layer_task->l->t_layer);
							prepend_executable(&layer_task_head, &layer_task_tail, n);
						}

						remove_executable(&layer_task_head, &layer_task_tail, layer_task);
						layer_task->done = true;
					}
					break;
				case EXECUTE_LINK:
					if (layer_task->l->counting_batch != batch) {
						mutate << <layer_task->l->size / thread_num + 1, layer_task->l->size>thread_num ? thread_num : layer_task->l->size, 0, streams[tasks] >> >(layer_task->l->dev_t.t, layer_task->l->dev_t.pr, layer_task->l->dev_t.po, gen);
						remove_executable(&layer_task_head, &layer_task_tail, layer_task);
						layer_task->done = true;
						tasks++;
						layer_task->l->mutating_batch = batch;
#ifdef DEBUG_SCHEDULE
						layer_t *s_phisical, *t_phisical, *s_logical, *t_logical;
						wrap_layers(layer_task, &s_phisical, &t_phisical, &s_logical, &t_logical);
						fprintf(stdout, "GEN:%d BATCH:%d JOB:%s FROM:%d TO:%d\n", gen, batch, "MUTW", s_logical->id, t_logical->id);
#endif
						remove_executable(&layer_task_head, &layer_task_tail, layer_task);
						layer_task->done = true;
					}
					break;
				case EXECUTE_LAYER:
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
							if (gen % 10 != 0) {
								integrate << <s_phisical->size / thread_num + 1, s_phisical->size>thread_num ? thread_num : s_phisical->size, 0, streams[tasks] >> > (s_phisical->dev_t[s_phisical->cur_s_dev_t].t, nullptr, nullptr, false);
							}
							else {
								integrate << <s_phisical->size / thread_num + 1, s_phisical->size>thread_num ? thread_num : s_phisical->size, 0, streams[tasks] >> > (s_phisical->dev_t[s_phisical->cur_s_dev_t].t, s_phisical->norm.t, s_phisical->lmbd.t, false);
							}
							layer_task->done = false;
							tasks++;
							s_phisical->integrated_gen = gen;

							append_updated_layer(&updated_layer_head, &updated_layer_tail, s_phisical);
						}

						link* pre = s_logical->pre;
						while (pre) {
							executable* n = new_executable(gen, EXECUTE_JOINT, pre->s_layer,  pre, s_logical);
							prepend_executable(&layer_task_head, &layer_task_tail, n);
							pre = pre->another_pre;
						}
						if (s_logical->logical_head && s_logical->logical_head->delegate) {
							link* pre = s_phisical->logical_head->pre;
							while (pre) {
								executable* n = new_executable(gen, EXECUTE_JOINT, pre->s_layer, pre, s_logical);
								prepend_executable(&layer_task_head, &layer_task_tail, n);
								pre = pre->another_pre;
							}
						}

						s_logical->integrated_gen = gen;
						s_logical->integrating_batch = batch;
#ifdef DEBUG_SCHEDULE
						fprintf(stdout, "GEN:%d BATCH:%d JOB:%s FROM:%d TO:%d BUFF:%d\n", gen, batch, "INTE", s_logical->id, t_logical->id, s_phisical->cur_s_dev_t);
#endif
					}
					else if (s_logical->integrating_batch != batch) {
						if (layer_task->l->mutating_batch != batch) {
							if (layer_task->l->pushed_gen != gen) {
								switch (layer_task->l->type) {
								case LINK_FORWARD:
									push_forward << <s_logical->size / thread_num + 1, s_logical->size>thread_num ? thread_num : s_logical->size, 0, streams[tasks] >> > (s_phisical->dev_t[s_phisical->cur_s_dev_t].t);
									break;
								case LINK_FULL:
									push_full << <s_logical->size / thread_num + 1, s_logical->size>thread_num ? thread_num : s_logical->size, 0, streams[tasks] >> > (s_phisical->dev_t[s_phisical->cur_s_dev_t].t, s_logical->offset, layer_task->l->dev_t.r, layer_task->l->dev_t.pr, s_logical->size, t_logical->size);
									break;
								}
								tasks++;
								layer_task->l->pushed_gen = gen;
#ifdef DEBUG_SCHEDULE
								fprintf(stdout, "GEN:%d BATCH:%d JOB:%s FROM:%d TO:%d BUFF:%d\n", gen, batch, "PUSH", s_logical->id, t_logical->id, s_phisical->cur_s_dev_t);
#endif
							}
							else {
								if (t_logical->pulling_gen != gen) {
									switch (layer_task->l->type) {
									case LINK_FORWARD:
										clear_pull_forward << <t_logical->size / thread_num + 1, t_logical->size>thread_num ? thread_num : t_logical->size, 0, streams[tasks] >> > (s_phisical->dev_t[s_phisical->cur_s_dev_t].t, t_phisical->dev_t[t_phisical->cur_t_dev_t].t, s_logical->offset, t_logical->offset, layer_task->l->dev_t.t, s_logical->size, gen);
										break;
									case LINK_FULL:
										if (!t_phisical->out) {
											clear_pull_full << <t_logical->size / thread_num + 1, t_logical->size>thread_num ? thread_num : t_logical->size, 0, streams[tasks] >> > (t_phisical->dev_t[t_phisical->cur_t_dev_t].t, t_phisical->norm.t, nullptr, nullptr, t_logical->offset, layer_task->l->dev_t.r, layer_task->l->dev_t.t, s_logical->size, gen, false);
										}
										else {
											clear_pull_full << <t_logical->size / thread_num + 1, t_logical->size>thread_num ? thread_num : t_logical->size, 0, streams[tasks] >> > (t_phisical->dev_t[t_phisical->cur_t_dev_t].t, t_phisical->norm.t, t_phisical->out_b.t, t_phisical->out_a.t, t_logical->offset, layer_task->l->dev_t.r, layer_task->l->dev_t.t, s_logical->size, gen, true);
										}
										break;
									}
									tasks++;
									remove_executable(&layer_task_head, &layer_task_tail, layer_task);
									layer_task->done = true;
									link* next_link = t_logical->next;
									while (next_link) {
										executable* n = new_executable(gen + 1, EXECUTE_LAYER, t_logical, next_link, next_link->t_layer);
										append_executable(&layer_task_head, &layer_task_tail, n);
										next_link = next_link->another_next;
									}
									if (t_logical != t_phisical) {
										if (t_phisical->pulling_gen != gen) {
											link* next_link = t_phisical->next;
											while (next_link) {
												executable* n = new_executable(gen + 1, EXECUTE_LAYER, t_phisical, next_link, next_link->t_layer);
												append_executable(&layer_task_head, &layer_task_tail, n);
												next_link = next_link->another_next;
											}
										}
									}
									else {
										layer_t* next = t_logical->logical_head;
										while (next) {
											if (!next->delegate) {
												link* next_link = next->next;
												while (next_link) {
													executable* n = new_executable(gen + 1, EXECUTE_LAYER, next, next_link, next_link->t_layer);
													append_executable(&layer_task_head, &layer_task_tail, n);
													next_link = next_link->another_next;
												}
											}
											next->pulling_gen = gen;
											next->pulling_batch = batch;
											next = next->next_logical;
										}
									}
									t_logical->pulling_gen = gen;
									t_logical->pulling_batch = batch;
									t_phisical->pulling_gen = gen;
									t_phisical->pulling_batch = batch;
#ifdef DEBUG_SCHEDULE
									fprintf(stdout, "GEN:%d BATCH:%d JOB:%s FROM:%d TO:%d BUFF:%d\n", gen, batch, "CPUL", s_logical->id, t_logical->id, t_phisical->cur_t_dev_t);
#endif
								}
								else if (t_logical->pulling_batch != batch) {
									switch (layer_task->l->type) {
									case LINK_FORWARD:
										pull_forward << <t_logical->size / thread_num + 1, t_logical->size>thread_num ? thread_num : t_logical->size, 0, streams[tasks] >> > (s_phisical->dev_t[s_phisical->cur_s_dev_t].t, t_phisical->dev_t[t_phisical->cur_t_dev_t].t, s_logical->offset, t_logical->offset, layer_task->l->dev_t.t, s_logical->size, gen);
										break;
									case LINK_FULL:
										if (!t_phisical->out) {
											pull_full << <t_logical->size / thread_num + 1, t_logical->size>thread_num ? thread_num : t_logical->size, 0, streams[tasks] >> > (t_phisical->dev_t[t_phisical->cur_t_dev_t].t, t_phisical->norm.t, nullptr, nullptr, t_logical->offset, layer_task->l->dev_t.r, layer_task->l->dev_t.t, s_logical->size, gen, false);
										}
										else {
											pull_full << <t_logical->size / thread_num + 1, t_logical->size>thread_num ? thread_num : t_logical->size, 0, streams[tasks] >> > (t_phisical->dev_t[t_phisical->cur_t_dev_t].t, t_phisical->norm.t, t_phisical->out_b.t, t_phisical->out_a.t, t_logical->offset, layer_task->l->dev_t.r, layer_task->l->dev_t.t, s_logical->size, gen, true);
										}
										break;
									}
									tasks++;
									remove_executable(&layer_task_head, &layer_task_tail, layer_task);
									layer_task->done = true;
									t_logical->pulling_batch = batch;
#ifdef DEBUG_SCHEDULE
									fprintf(stdout, "GEN:%d BATCH:%d JOB:%s FROM:%d TO:%d BUFF:%d\n", gen, batch, "PULL", s_logical->id, t_logical->id, t_phisical->cur_t_dev_t);
#endif
								}
							}
						}
					}
					break;
				}
			}
			else {
				break;
			}
			executable* tmp = layer_task;
			layer_task = layer_task->next;
			if (tmp->done) {
				free(tmp);
			}
		}
		for (int i = 0; i < tasks; i++) {
			cudaStreamSynchronize(streams[i]);
		}
		//critical region end
		layer_t* updated_layer = updated_layer_head;
		while (updated_layer) {
#ifdef DEBUG_DATA
				cudaMemcpy(updated_layer->host_t.t, updated_layer->dev_t[updated_layer->cur_s_dev_t].t, updated_layer->size * sizeof(float), cudaMemcpyDeviceToHost);
				fprintf(stdout, "LAYER[%d]:{", updated_layer->id);
				for (int i = 0; i < updated_layer->size; i++) {
					fprintf(stdout, " %f,", updated_layer->host_t.t[i]);
				}
				fprintf(stdout, " }\n");
#endif // DEBUG_DATA
				layer_t* tmp = updated_layer->updated_next;
				remove_updated_layer(&updated_layer_head, &updated_layer_tail, updated_layer);
				updated_layer = tmp;
		}
	}
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

	construct_network();

	init_network();

	execute(200);

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
