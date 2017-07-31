function class = knn_score(train, test, Kmax, train_label)

dis = sum((train-test*ones(1, size(train, 2))).^2);
[~, idx] = sort(dis, 'ascend');
idx = idx(1:Kmax);
min_lbl = train_label(idx);
qq = zeros(Kmax, max(min_lbl));

qq((1:Kmax)'+(min_lbl-1)*Kmax) = 1;
qq = cumsum(qq, 1);
[~, class] = max(qq, [], 2);