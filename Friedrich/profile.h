#pragma once

#define DEBUG_SCHEDULE
#define DEBUG_DATA

//#define PROFILE_FORMAL
#define PROFILE_TEST_NEURON
//#define PROFILE_TEST_SYNAPASE
//#define PROFILE_TEST_INTEGRATE
//#define PROFILE_TEST_MUTATE
//#define PROFILE_TEST_NEURON2

#ifdef PROFILE_FORMAL
#include "profile_formal.h"
#endif // PROFILE_FORMAL

#ifdef PROFILE_TEST_NEURON
#include "profile_test_neuron.h"
#endif // PROFILE_TEST_NEURON

#ifdef PROFILE_TEST_SYNAPASE
#include "profile_test_synapase.h"
#endif // PROFILE_TEST_SYNAPASE

#ifdef PROFILE_TEST_INTEGRATE
#include "profile_test_integrate.h"
#endif // PROFILE_TEST_INTEGRATE

#ifdef PROFILE_TEST_MUTATE
#include "profile_test_mutate.h"
#endif // PROFILE_TEST_MUTATE

#ifdef PROFILE_TEST_NEURON2
#include "profile_test_neuron2.h"
#endif // PROFILE_TEST_NEURON2