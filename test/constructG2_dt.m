G2unitdt=zeros(9999,9999);
for i=0:9998
filename=strcat('binary/G2unit_dt',int2str(i),'.bin')
file=fopen(filename);
G2unitdt(:,i+1)=fread(file,[1,9999],'double');
frewind(file);
fclose(file);
end