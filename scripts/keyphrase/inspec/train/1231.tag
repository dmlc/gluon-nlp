efficient	NO_KP
parallel	BEGIN_KP
programming	INSIDE_KP
on	NO_KP
scalable	BEGIN_KP
shared	INSIDE_KP
memory	INSIDE_KP
systems	NO_KP
with	NO_KP
high	BEGIN_KP
performance	INSIDE_KP
fortran	NO_KP
openmp	NO_KP
offers	NO_KP
a	NO_KP
high-level	NO_KP
interface	NO_KP
for	NO_KP
parallel	BEGIN_KP
programming	INSIDE_KP
on	NO_KP
scalable	BEGIN_KP
shared	INSIDE_KP
memory	INSIDE_KP
(smp)	NO_KP
architectures.	NO_KP
it	NO_KP
provides	NO_KP
the	NO_KP
user	NO_KP
with	NO_KP
simple	NO_KP
work-sharing	NO_KP
directives	NO_KP
while	NO_KP
it	NO_KP
relies	NO_KP
on	NO_KP
the	NO_KP
compiler	NO_KP
to	NO_KP
generate	NO_KP
parallel	BEGIN_KP
programs	NO_KP
based	NO_KP
on	NO_KP
thread	NO_KP
parallelism.	NO_KP
however,	NO_KP
the	NO_KP
lack	NO_KP
of	NO_KP
language	NO_KP
features	NO_KP
for	NO_KP
exploiting	NO_KP
data	NO_KP
locality	NO_KP
often	NO_KP
results	NO_KP
in	NO_KP
poor	NO_KP
performance	BEGIN_KP
since	NO_KP
the	NO_KP
non-uniform	NO_KP
memory	BEGIN_KP
access	NO_KP
times	NO_KP
on	NO_KP
scalable	BEGIN_KP
smp	NO_KP
machines	NO_KP
cannot	NO_KP
be	NO_KP
neglected.	NO_KP
high	BEGIN_KP
performance	INSIDE_KP
fortran	NO_KP
(hpf),	NO_KP
the	NO_KP
de-facto	NO_KP
standard	NO_KP
for	NO_KP
data	NO_KP
parallel	BEGIN_KP
programming,	NO_KP
offers	NO_KP
a	NO_KP
rich	NO_KP
set	NO_KP
of	NO_KP
data	NO_KP
distribution	NO_KP
directives	NO_KP
in	NO_KP
order	NO_KP
to	NO_KP
exploit	NO_KP
data	NO_KP
locality,	NO_KP
but	NO_KP
it	NO_KP
has	NO_KP
been	NO_KP
mainly	NO_KP
targeted	NO_KP
towards	NO_KP
distributed	NO_KP
memory	BEGIN_KP
machines.	NO_KP
in	NO_KP
this	NO_KP
paper	NO_KP
we	NO_KP
describe	NO_KP
an	NO_KP
optimized	NO_KP
execution	NO_KP
model	NO_KP
for	NO_KP
hpf	NO_KP
programs	NO_KP
on	NO_KP
smp	NO_KP
machines	NO_KP
that	NO_KP
avails	NO_KP
itself	NO_KP
with	NO_KP
mechanisms	NO_KP
provided	NO_KP
by	NO_KP
openmp	NO_KP
for	NO_KP
work	NO_KP
sharing	NO_KP
and	NO_KP
thread	NO_KP
parallelism,	NO_KP
while	NO_KP
exploiting	NO_KP
data	NO_KP
locality	NO_KP
based	NO_KP
on	NO_KP
user-specified	NO_KP
distribution	NO_KP
directives.	NO_KP
data	NO_KP
locality	NO_KP
does	NO_KP
not	NO_KP
only	NO_KP
ensure	NO_KP
that	NO_KP
most	NO_KP
memory	BEGIN_KP
accesses	NO_KP
are	NO_KP
close	NO_KP
to	NO_KP
the	NO_KP
executing	NO_KP
threads	NO_KP
and	NO_KP
are	NO_KP
therefore	NO_KP
faster,	NO_KP
but	NO_KP
it	NO_KP
also	NO_KP
minimizes	NO_KP
synchronization	NO_KP
overheads,	NO_KP
especially	NO_KP
in	NO_KP
the	NO_KP
case	NO_KP
of	NO_KP
unstructured	NO_KP
reductions.	NO_KP
the	NO_KP
proposed	NO_KP
shared	BEGIN_KP
memory	INSIDE_KP
execution	NO_KP
model	NO_KP
for	NO_KP
hpf	NO_KP
relies	NO_KP
on	NO_KP
a	NO_KP
small	NO_KP
set	NO_KP
of	NO_KP
language	NO_KP
extensions,	NO_KP
which	NO_KP
resemble	NO_KP
the	NO_KP
openmp	NO_KP
work-sharing	NO_KP
features.	NO_KP
these	NO_KP
extensions,	NO_KP
together	NO_KP
with	NO_KP
an	NO_KP
optimized	NO_KP
shared	BEGIN_KP
memory	INSIDE_KP
parallelization	NO_KP
and	NO_KP
execution	NO_KP
model,	NO_KP
have	NO_KP
been	NO_KP
implemented	NO_KP
in	NO_KP
the	NO_KP
adaptor	NO_KP
hpf	NO_KP
compilation	NO_KP
system	NO_KP
and	NO_KP
experimental	NO_KP
results	NO_KP
verify	NO_KP
the	NO_KP
efficiency	NO_KP
of	NO_KP
the	NO_KP
chosen	NO_KP
approach	NO_KP
