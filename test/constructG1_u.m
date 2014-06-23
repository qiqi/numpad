G1unitu=zeros(20000,20000);
for i=1:10000
filename=strcat('binary/G1unit_u1_',int2str(i-1),'.bin')
file=fopen(filename);
G1unitu(:,2*i-1)=fread(file,[1,20000],'double');
frewind(file);
fclose(file);

filename=strcat('binary/G1unit_u2_',int2str(i-1),'.bin')
file=fopen(filename);
G1unitu(:,2*i)=fread(file,[1,20000],'double');
frewind(file);
fclose(file);
end