function [Psi, Phi, X, Record] = ParsevalKSVD(Y, Psi0, Phi0, X0, maxIter, t, rho, IsRecord, ShowDetail)
%2019/6/4
%author: B.C. Kung

%For record
augLag = zeros(1, maxIter);
obj_rep = zeros(1, maxIter);
obj_total = zeros(1, maxIter);
con1 = zeros(1, maxIter);
con2 = zeros(1, maxIter);

%initial
[m, n] = size(Psi0);
Psi = Psi0;
Phi = Phi0;
X = X0;
lam2 = zeros(m);
lam3 = zeros(m, n);


for i = 1 : maxIter
    for inner_ind = 1 : 5
        %updae Psi (frame)
        Psi = oneColumnPsi(Phi, Psi, X, Y, lam2, lam3, rho, t, 1, 2);
       
        % update Phi (dual frame)
        Phi = oneColumnPhi(Phi, Psi, X, Y, lam2, lam3, rho, t, 1, 2);
        
        %update lambda2
        lam2 = lam2 + rho(2)*(Psi*(Phi)' - eye(m));
        
        %update lambda3
        lam3 = lam3 + rho(3)*(Psi - Phi);
    end
    %update X
    X = upateX(X, Y, Psi, Phi, rho(1));
    
    if(IsRecord) %If need record the detail
        augLag(i) = rho(1)*norm(Y - Psi*X, 'fro')^2 + norm( (Phi')*Y - (Phi')*Psi*X, 'fro')^2 ...
        + trace((lam2')*(Psi*(Phi') - eye(m))) + (rho(2)/2)*norm(Psi*(Phi') - eye(m), 'fro')^2 ...
        + trace((lam3')*(Psi - Phi)) + (rho(3)/2)*norm(Psi - Phi, 'fro')^2;
        obj_rep(i) = norm(Y - Psi*X, 'fro')^2;
        obj_total(i) = rho(1)*norm(Y - Psi*X, 'fro')^2 + norm((Phi')*Y - (Phi')*Psi*X, 'fro')^2;
        con1(i) = norm(Psi*(Phi') - eye(m), 'fro')^2;
        con2(i) = norm(Psi - Phi, 'fro')^2;
    end
    if(ShowDetail)
        %show detail
        disp(['Iteration = ', num2str(i)]);
        fprintf('L=%-10.4g, obj_rep=%-10.4g, obj_total=%-10.4g, con1=%-10.4g, con2=%-10.4g\n',...
        augLag(i), obj_rep(i), obj_total(i), con1(i), con2(i));
    end
end
Record.augLag = augLag;
Record.obj_rep = obj_rep;
Record.obj_total = obj_total;
Record.con1 = con1;
Record.con2 = con2;

function [newX] = upateX(X, Y, Psi, Phi, rho1)
%version: 2019/6/4
%update X by row wise

newX = X;
for k = 1 : size(X,1)
    gk = []; %gk is the non-zero entries index in column vector X(k,:)
    for j = 1 : size(X,2)
        if abs(X(k, j))>=1.0e-20
            gk = [gk, j];
        end
    end
    Gk = zeros(size(X,2) ,length(gk)); %Multiply Wk on left hand side to extract the non-zeros component in X(k,:)
    for j = 1 : length(gk)
        Gk(gk(j),j) = 1;
    end
    %create Error Matrix
    fullind = 1:size(X,1);
    B = (Phi')*Y;
    Ek = Y - Psi(:, setdiff(fullind,k))*newX(setdiff(fullind,k),:);
    Fk = B - (Phi')*Psi(:, setdiff(fullind,k))*newX(setdiff(fullind,k),:);
    Ek_tilde = Ek*Gk; %restrained Ek
    Fk_tilde = Fk*Gk; %restrained Fk
    Psi_k = Psi(:, k);
    new_Xk = (rho1*(Psi_k')*Ek_tilde + (Psi_k')*Phi*Fk_tilde)/(rho1*norm(Psi_k, 2)^2 + norm((Phi')*Psi_k, 2)^2);
    newX(k, :) = new_Xk*Gk';
end

return;

function [newpsi, objValue, LagValue] = oneColumnPsi(Phi, Psi, X, Y, lam2, lam3, rho, t, iter1, iter2)
%author: BC. Kung
%version: 2019/6/3
%iter1: the number of column loop
%iter2: iteration number of the projection gradient descent
%t: projection gradient descent step size

[m, n] = size(Psi);
rho1 = rho(1);
rho2 = rho(2);
rho3 = rho(3);

%Obj is objective function
Obj = @(v_psi) norm(Y - v_psi*X,'fro')^2;
%L is augmented lagrange function
L = @(v_psi) rho1*norm(Y - v_psi*X, 'fro')^2 + norm((Phi')*Y - (Phi')*v_psi*X, 'fro')^2 + ...
    trace((lam2')*(v_psi*(Phi') - eye(m))) + (rho2/2)*norm(v_psi*(Phi') - eye(m), 'fro')^2 + ...
    trace((lam3')*(v_psi - Phi)) + (rho3/2)*norm(v_psi - Phi, 'fro')^2;


objValue = zeros(1, iter1*iter2*n+1);
LagValue = zeros(1, iter1*iter2*n+1);

objValue(1) = Obj(Psi);
LagValue(1) = L(Psi);

ind = 2;
for j = 1 : iter1
    for k = 1 : n
        psia = Psi(:,1:(k-1));
        psib = Psi(:,(k+1):n);
        phia = Phi(:,1:(k-1));
        phik = Phi(:,k);
        phib = Phi(:,(k+1):n);
        Xa = X(1:(k-1), :);
        Xk = X(k,:);
        Xb = X((k+1):n,:);
        lam3k = lam3(:, k);
        c1 = 2*rho1*eye(m) + 2*Phi*(Phi');
        c2 = rho2*(phik')*phik + rho3;
        c3 = -2*rho1*Y*(Xk') + 2*rho1*psia*Xa*(Xk') +2*rho1*psib*Xb*(Xk') - 2*Phi*(Phi')*Y*(Xk') ...
            + 2*Phi*(Phi')*psia*Xa*(Xk') + 2*Phi*(Phi')*psib*Xb*(Xk') + lam2*phik + rho2*psia*(phia')*phik ...
            + rho2*psib*(phib')*phik - rho2*phik + lam3k - rho3*phik;
        
        for l = 1 : iter2 %projection gradient descent loop
            dL = c1*Psi(:,k)*Xk*(Xk') + Psi(:,k)*c2 + c3;
            newPsik = Psi(:,k) - t*dL;
            newPsik = newPsik/norm(newPsik);
            Psi = [psia, newPsik, psib];
            objValue(ind) = Obj(Psi);
            LagValue(ind) = L(Psi);
            ind = ind + 1;
        end

    end
end
newpsi = Psi;

function [newphi, LagValue] = oneColumnPhi(Phi, Psi, X, Y, lam2, lam3, rho, t, iter1, iter2)
%author: BC. Kung
%version: 2019/6/3
%iter1: the number of column loop
%iter2: iteration number of the projection gradient descent
%t: projection gradient descent step size

[m, n] = size(Phi);
rho1 = rho(1);
rho2 = rho(2);
rho3 = rho(3);

%L is objective function
L = @(v_phi) rho1*norm(Y - Psi*X, 'fro')^2 + norm((v_phi')*Y - (v_phi')*Psi*X, 'fro')^2 ...
    + trace((lam2')*(Psi*(v_phi') - eye(m))) + (rho2/2)*norm(Psi*(v_phi') - eye(m), 'fro')^2 ...
    + trace((lam3')*(Psi - v_phi)) + (rho3/2)*norm(Psi - v_phi, 'fro')^2;


LagValue = zeros(1, iter1*iter2*n+1);

LagValue(1) = L(Phi);

ind = 2;
for j = 1 : iter1
    for k = 1 : n
        psia = Psi(:,1:(k-1));
        psik = Psi(:,k);
        psib = Psi(:,(k+1):n);
        phia = Phi(:,1:(k-1));
        phib = Phi(:,(k+1):n);
        lam3k = lam3(:, k);
        
        c1 = 2*Y*(Y') - 2*Y*(X')*(Psi') - 2*Psi*X*(Y') + 2*Psi*X*(X')*(Psi');
        c2 = rho2*(psik')*psik + rho3;
        c3 = (lam2')*psik + rho2*phia*(psia')*psik + rho2*phib*(psib')*psik - rho2*psik - lam3k - rho3*psik;
        
        for l = 1 : iter2 %projection gradient descent loop
            dL = c1*Phi(:,k) + Phi(:,k)*c2 + c3;
            newPhik = Phi(:,k) - t*dL;
            newPhik = newPhik/norm(newPhik);
            Phi = [phia, newPhik, phib];
            LagValue(ind) = L(Phi);
            ind = ind + 1;
        end
    end
end
newphi = Phi;