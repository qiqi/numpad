A=zeros(29999,29999);
'block 1'
A(1:20000,1:20000) = G1unitu;
'block 2'
A(1:20000,20001:29999) = G1unitdt';
'block 3'
A(20001:29999,1:20000) = G2unitu';
'block 4'
A(20001:29999,20001:29999) = G2unitdt;

'Eigenvalues'
eigenvalues = eig(A);