A=zeros(14,14);
'block 1'
A(1:10,1:10) = G1unitu;
'block 2'
A(1:10,11:14) = G1unitdt;
'block 3'
A(11:14,1:10) = G2unitu;
'block 4'
A(11:14,11:14) = G2unitdt;

'Eigenvalues'
[U,V] = eig(A);