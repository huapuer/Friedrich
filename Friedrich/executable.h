#pragma once

#include "../../Ludwig/Ludwig/ludwig_neural_network.h"

enum execute_type {
	EXECUTE_LAYER,
	EXECUTE_LINK
};

struct executable {
	unsigned long long gen;
	execute_type type;
	layer_t* s;
	link* l;
	layer_t* t;
	executable* pre;
	executable* next;
	bool done;
};

executable* new_executable(int gen, execute_type type, layer_t* s, link* l, layer_t* t) {
	executable* ret = (executable*)malloc(sizeof(executable));
	memset(ret, 0, sizeof(executable));
	ret->gen = gen;
	ret->type = type;
	ret->s = s;
	ret->l = l;
	ret->t = t;
	ret->done = false;
	return ret;
}

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

void prepend_executable(executable** head, executable** tail, executable* n) {
	if (!*head) {
		*head = n;
		*tail = n;
	}
	else {
		(*head)->pre = n;
		(*head)->pre->next = *head;
		*head = (*head)->pre;
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