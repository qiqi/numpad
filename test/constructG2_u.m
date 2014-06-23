G2unitu=zeros(20000,9999);
for i=0:9998
filename=strcat('binary/G2unit_u',int2str(i),'.bin')
file=fopen(filename);
G2unitu(:,i+1)=fread(file,[1,20000],'double');
frewind(file);
fclose(file);
end