function [pos_ind] = get_lebedev_ind(posin,Q)
[positions_sparse, ~, ~] = supdeq_lebedev(0,Q);
posin(:,2) = 90 - posin(:,2);

pos_ind = zeros(1,size(positions_sparse,1));
for i = 1:size(positions_sparse,1)
    dis = abs(positions_sparse(i,1) - posin(:,1)) + abs(positions_sparse(i,2) - posin(:,2));
    [~,ind] = sort(dis,'ascend');
    for j = 1:length(posin)
        if ~ismember(ind(j),pos_ind)
            pos_ind(1,i) = ind(j);
            break;
        end
    end
end
end

