function [ c ] = cost(f,u,gamma)
%COST Summary of this function goes here
%   Detailed explanation goes here
M=size(u,1);
N=size(u,2);
c=sum(sum(sum((u-f).^2)));
for i=1:M
    for j=1:N
        if sum(u(i,j,:)~=u(min(i+1,M),j,:))>0
            c=c+gamma;
        end
        if sum(u(i,j,:)~=u(i,min(j+1,N),:))>0
            c=c+gamma;
        end
    end
end
end

