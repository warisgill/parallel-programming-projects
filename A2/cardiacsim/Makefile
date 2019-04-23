include ./arch.gnu
# OPTIMIZATION = -fast
# OPTIMIZATION = -O3
# DEBUG += -g


app:		serial_cardiacsim d1_cardiacsim d2_cardiacsim openmp_d2_cardiacsim 

OBJECT0 = serial_cardiacsim.o splot.o cmdLine.o
OBJECT1 = d1_cardiacsim.o splot.o cmdLine.o
OBJECT2 = d2_cardiacsim.o splot.o cmdLine.o
OBJECT3 = openmp_d2_cardiacsim.o splot.o cmdLine.o


serial_cardiacsim:	        $(OBJECT0) 
		$(C++LINK) $(LDFLAGS) -o $@ $(OBJECT0)  $(LDLIBS)

d1_cardiacsim:	        $(OBJECT1) 
		$(C++LINK) $(LDFLAGS) -o $@ $(OBJECT1)  $(LDLIBS)

d2_cardiacsim:	        $(OBJECT2) 
		$(C++LINK) $(LDFLAGS) -o $@ $(OBJECT2)  $(LDLIBS)

openmp_d2_cardiacsim:	        $(OBJECT3) 
		$(C++LINK) $(LDFLAGS) -o $@ $(OBJECT3)  $(LDLIBS)


clean:	
	$(RM) *.o a.out openmp_d2_cardiacsim d2_cardiacsim d1_cardiacsim serial_cardiacsim *~;
	$(RM) core;


