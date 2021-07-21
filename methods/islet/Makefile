include make.inc

SOURCES = isl.cpp tables.cpp npx.cpp maxeigcomp.cpp searchxnodes.cpp pum.cpp studymetrics.cpp nodalbasis.cpp np4.cpp
OBJECTS = $(SOURCES:.cpp=.o)
.cpp.o:
	$(CXX) $(CXXFLAGS) -c $< -o $@

all: libislet search np4 pum_sweep run_meam1_sweep cslunstab

libislet: $(OBJECTS)
	$(CXX) $(OBJECTS) $(LDFLAGS) -fopenmp -shared $(LINK_LAPACK_BLAS) -o libislet.so

search: libislet search.o
	$(CXX) search.o $(LDFLAGS) $(LINK_LAPACK_BLAS) -L. -lislet -fopenmp -o search

np4: libislet run_np4.o
	$(CXX) run_np4.o $(LDFLAGS) $(LINK_LAPACK_BLAS) -L. -lislet -fopenmp -o np4

pum_perturb_plot: libislet pum_perturb_plot.o
	$(CXX) pum_perturb_plot.o $(LDFLAGS) $(LINK_LAPACK_BLAS) -L. -lislet -fopenmp -o pum_perturb_plot

run_meam1_sweep: libislet run_meam1_sweep.o
	$(CXX) run_meam1_sweep.o $(LDFLAGS) $(LINK_LAPACK_BLAS) -L. -lislet -fopenmp -o run_meam1_sweep

pum_sweep: libislet pum_sweep.o
	$(CXX) pum_sweep.o $(LDFLAGS) $(LINK_LAPACK_BLAS) -L. -lislet -fopenmp -o pum_sweep

cslunstab: cslunstab.o
	$(CXX) cslunstab.o $(LDFLAGS) $(LINK_LAPACK_BLAS) -fopenmp -o cslunstab

clean:
	rm -f *.o *.so search np4 pum_sweep pum_perturb_plot run_meam1_sweep

search: search.cpp
cslunstab: cslunstab.cpp
maxeigcomp.o: maxeigcomp.hpp npx.hpp interpmethod.hpp
