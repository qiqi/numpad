G1unitdt=zeros(10,4);
for i=1:5
i
filename=strcat('../binaryN5/G1unit_dt1_',int2str(i-1),'.bin')
file=fopen(filename);
G1unitdt(2*i-1,:)=fread(file,[1,4],'double');
frewind(file);
fclose(file);

filename=strcat('../binaryN5/G1unit_dt2_',int2str(i-1),'.bin')
file=fopen(filename);
G1unitdt(2*i,:)=fread(file,[1,4],'double');
frewind(file);
fclose(file);
end