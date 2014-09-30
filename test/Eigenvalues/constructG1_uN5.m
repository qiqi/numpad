G1unitu=zeros(10,10);
for i=1:5
filename=strcat('../binaryN5/G1unit_u1_',int2str(i-1),'.bin')
file=fopen(filename);
G1unitu(2*i-1,:)=fread(file,[1,10],'double');
frewind(file);
fclose(file);

filename=strcat('../binaryN5/G1unit_u2_',int2str(i-1),'.bin')
file=fopen(filename);
G1unitu(2*i,:)=fread(file,[1,10],'double');
frewind(file);
fclose(file);
end