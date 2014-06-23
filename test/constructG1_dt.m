G1unitdt=zeros(9999,20000);
for i=1:10000
i
filename=strcat('binary/G1unit_dt1_',int2str(i-1),'.bin')
file=fopen(filename);
G1unitdt(:,2*i-1)=fread(file,[1,9999],'double');
frewind(file);
fclose(file);

filename=strcat('binary/G1unit_dt2_',int2str(i-1),'.bin')
file=fopen(filename);
G1unitdt(:,2*i)=fread(file,[1,9999],'double');
frewind(file);
fclose(file);
end