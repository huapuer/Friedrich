#pragma once

//#define PROFILE_FORMAL
//#define PROFILE_TEST_NEURON
#define PROFILE_TEST_SYNAPASE

#ifdef PROFILE_FORMAL
#include "profile_formal.h"
#endif // PROFILE_FORMAL

#ifdef PROFILE_TEST_NEURON
#include "profile_test_neuron.h"
#endif // PROFILE_TEST_NEURON

#ifdef PROFILE_TEST_SYNAPASE
#include "profile_test_synapase.h"
#endif // PROFILE_TEST_SYNAPASE