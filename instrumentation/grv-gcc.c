/*
  Copyright 2013 Google LLC All rights reserved.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at:

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

/*
   american fuzzy lop - wrapper for GCC and clang
   ----------------------------------------------

   Written and maintained by Michal Zalewski <lcamtuf@google.com>

   This program is a drop-in replacement for GCC or clang. The most common way
   of using it is to pass the path to afl-gcc or afl-clang via CC when invoking
   ./configure.

   (Of course, use CXX and point it to afl-g++ / afl-clang++ for C++ code.)

   The wrapper needs to know the path to afl-as (renamed to 'as'). The default
   is /usr/local/lib/afl/. A convenient way to specify alternative directories
   would be to set GRV_PATH.

   If AFL_HARDEN is set, the wrapper will compile the target app with various
   hardening options that may help detect memory management issues more
   reliably. You can also specify AFL_USE_ASAN to enable ASAN.

   If you want to call a non-default compiler as a next step of the chain,
   specify its location via AFL_CC or AFL_CXX.

*/

/*
  Edited by iman <shz230@nyu.edu>
*/

#define AFL_MAIN

#include "config.h"
#include "types.h"
#include "debug.h"
#include "alloc-inl.h"

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

static u8*  as_path;                /* Path to the GRV 'as' wrapper      */
static u8** cc_params;              /* Parameters passed to the real CC  */
static u32  cc_par_cnt = 1;         /* Param count, including argv0      */
static u8   be_quiet;               /* Quiet mode                        */


/* Try to find our "fake" GNU assembler in GRV_PATH or at the location derived
   from argv[0]. If that fails, abort. */

static void find_as(u8* argv0) {

  u8 *grv_path = getenv("GRV_PATH");
  u8 *slash, *tmp;

  if (grv_path) {

    tmp = alloc_printf("%s/as", grv_path);

    if (!access(tmp, X_OK)) {
      as_path = grv_path;
      ck_free(tmp);
      return;
    }

    ck_free(tmp);

  }

  slash = strrchr(argv0, '/');

  if (slash) {

    u8 *dir;

    *slash = 0;
    dir = ck_strdup(argv0);
    *slash = '/';

    tmp = alloc_printf("%s/grv-as", dir);

    if (!access(tmp, X_OK)) {
      as_path = dir;
      ck_free(tmp);
      return;
    }

    ck_free(tmp);
    ck_free(dir);

  }

  if (!access(GRV_PATH "/as", X_OK)) {
    as_path = GRV_PATH;
    return;
  }

  FATAL("Unable to find GRV wrapper binary for 'as'. Please set GRV_PATH");
 
}


/* Copy argv to cc_params, making the necessary edits. */

static void edit_params(u32 argc, char** argv) {

  u8 *name;
  cc_params = ck_alloc((argc + 128) * sizeof(u8*));

  name = strrchr(argv[0], '/');
  if (!name) name = argv[0]; else name++;

    if (!strcmp(name, "grv-g++")) {
      u8* alt_cxx = getenv("GRV_CXX");
      cc_params[0] = alt_cxx ? alt_cxx : (u8*)"riscv64-unknown-elf-g++";
    } else {
      u8* alt_cc = getenv("GRV_CC");
      cc_params[0] = alt_cc ? alt_cc : (u8*)"riscv64-unknown-elf-gcc";
    }

  while (--argc) {
    u8* cur = *(++argv);
    if (!strncmp(cur, "-B", 2)) {
      if (!be_quiet) WARNF("-B is already set, overriding");
      if (!cur[2] && argc > 1) { argc--; argv++; }
      continue;
    }
    cc_params[cc_par_cnt++] = cur;
  }

  cc_params[cc_par_cnt++] = "-B";
  cc_params[cc_par_cnt++] = as_path;

  if (!getenv("GRV_DONT_OPTIMIZE")) {
    cc_params[cc_par_cnt++] = "-O3";
    cc_params[cc_par_cnt++] = "-funroll-loops";
    /* Two indicators that you're building for fuzzing; one of them is
       AFL-specific, the other is shared with libfuzzer. */
    cc_params[cc_par_cnt++] = "-D__AFL_COMPILER=1";
    cc_params[cc_par_cnt++] = "-DFUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION=1";
  }

  // if (getenv("GRV_NO_BUILTIN")) {
  //   cc_params[cc_par_cnt++] = "-fno-builtin-strcmp";
  //   cc_params[cc_par_cnt++] = "-fno-builtin-strncmp";
  //   cc_params[cc_par_cnt++] = "-fno-builtin-strcasecmp";
  //   cc_params[cc_par_cnt++] = "-fno-builtin-strncasecmp";
  //   cc_params[cc_par_cnt++] = "-fno-builtin-memcmp";
  //   cc_params[cc_par_cnt++] = "-fno-builtin-strstr";
  //   cc_params[cc_par_cnt++] = "-fno-builtin-strcasestr";
  // }
  cc_params[cc_par_cnt] = NULL;
}


/* Main entry point */

int main(int argc, char** argv) {

  if (isatty(2) && !getenv("GRV_QUIET")) {

    SAYF(cCYA "grv-cc " cBRI VERSION cRST " by <lcamtuf@google.com>\n");

  } else be_quiet = 1;

  if (argc < 2) {

    SAYF("\n"
         "This is a helper application for grv. It serves as a drop-in replacement\n"
         "for gcc letting you recompile third-party code with the required\n"
         "runtime instrumentation. A common use pattern would be one of the following:\n\n"

         "  CC=%s/grv-gcc ./configure\n"
         "  CXX=%s/grv-g++ ./configure\n\n"

         "You can specify custom next-stage toolchain via GRV_CC, GRV_CXX, and GRV_AS.\n",
         BIN_PATH, BIN_PATH);

    exit(1);
  }

  find_as(argv[0]);
  printf("FOUND AS in: %s\n", as_path);
  edit_params(argc, argv);
  printf("execvp.. (%s)\n", cc_params[0]);
  execvp(cc_params[0], (char**)cc_params);

  FATAL("Oops, failed to execute '%s' - check your PATH", cc_params[0]);

  return 0;

}