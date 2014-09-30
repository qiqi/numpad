G2unitu=zeros(4,10);
for i=0:3
filename=strcat('../binaryN5/G2unit_u',int2str(i),'.bin')
file=fopen(filename);
G2unitu(i+1,:)=fread(file,[1,10],'double');
frewind(file);
fclose(file);
end