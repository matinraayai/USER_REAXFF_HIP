#  Makefile template for Static library. 
# 1. Compile every *.cpp in the folder 
# 2. All obj files under obj folder
# 3. static library .a at lib folder
# 4. run 'make dirmake' before calling 'make'



EXTRAMAKE = Makefile.lammps.empty


HIPCC = /opt/rocm/bin/hipcc

LINK =		hipcc
LINKFLAGS =	-g -O3  $(shell mpicxx --showme:compile) -DHAVE_HIP

CC = hipcc

HIPCCFLAGS  =  -w -g -O3 -DHAVE_HIP -DLAMMPS_REAX -MP

HIPCCINCS = $(shell mpicxx --showme:compile)

HIPCCDEFS = $(HIPCCFLAGS) $(HIPCCINCS)

CCINCS =  -g -O3  $(shell mpicxx --showme:compile) -DHAVE_HIP -DLAMMPS_REAX  -MP

CCFLAGS = -w
CCDEFS =  $(CCFLAGS) $(CCINCS)

OUT_FILE_NAME = libreaxcgpuhip.a

OBJ_DIR=./obj

OUT_DIR=./

# Enumerating of every *.cpp as *.o and using that as dependency.	
# filter list of .c files in a directory.
# FILES =dump_l.c \
#	kter.c \
#
# $(OUT_FILE_NAME): $(patsubst %.c,$(OBJ_DIR)/%.o,$(wildcard $(FILES))) 


# Enumerating of every *.cpp as *.o and using that as dependency
$(OUT_FILE_NAME): $(patsubst %.cu,$(OBJ_DIR)/%.o,$(wildcard *.cu))
	ar -rcs -o $(OUT_DIR)/$@ $^

INCLUDE_FILES_CPP=./common

$(OUT_FILE_NAME): $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(wildcard *.cpp))
	ar -rcs -o $(OUT_DIR)/$@ $^

build:
	$(OBJ_DIR)/%.o $(OBJ_DIR)/%.o

#Compiling every *.cpp to *.o
$(OBJ_DIR)/%.o: %.cu dirmake
	$(HIPCC) -c $(HIPCCDEFS) -I$(INCLUDE_FILES_CPP) -o $@  $<

$(OBJ_DIR)/%.o: %.cpp dirmake
	$(CC) -c $(CCDEFS) -o $@  $<
dirmake:
	@mkdir -p $(OUT_DIR)
	@mkdir -p $(OBJ_DIR)
	@cp $(EXTRAMAKE) Makefile.lammps

	
clean:
	rm -f $(OBJ_DIR)/*.o $(OUT_DIR)/$(OUT_FILE_NAME) Makefile.bak

rebuild: clean build


 
