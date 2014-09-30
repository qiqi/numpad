G2unitdt=zeros(4,4);
for i=0:3
filename=strcat('../binaryN5/G2unit_dt',int2str(i),'.bin')
file=fopen(filename);
G2unitdt(i+1,:)=fread(file,[1,4],'double');
frewind(file);
fclose(file);
end