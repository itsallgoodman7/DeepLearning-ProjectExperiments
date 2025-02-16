function [A,B,R,err,c]=retropro(Inp,Targ,nhid,Eta,Err,Nmax,seme);
%
tic
%
[N,n]=size(Inp); % n num neuroni di input
[N1,z]=size(Targ); % z num neuroni uscita (target)

if N~=N1 
fprintf('Dimensionamento non corretto\n\n');
% break;
end

%Inizializzazione random dei pesi fra -1 e 1
rand('state',seme);
A=2*(rand(nhid,n)-0.5*ones(nhid,n)); % pesi di primo livello (quelli fra input e hidden)
B=2*(rand(z,nhid)-0.5*ones(z,nhid)); % pesi di secondo livello (quelli fra hidden e uscita)

Inp=Inp';
Targ=Targ';
Nd=N*z;
err=[];

c=0;
ciclo=0;

while ciclo==0
    R=f(B*f(A*Inp)); % step forward
	q=ones(1,z)*((R-Targ).^2)*ones(N,1)/Nd; % errore quadratico medio (sommatoria da 1 a n)
        err=[err q]; % contiene tutti gli errori q

	if q<=Err | c>=Nmax % controllo soglia errore e c inferiore a num.max iterazioni
		ciclo=1;
	end
	
	if ciclo==0
		c=c+1;
		for k=1:N
			%
			% Modifica di A e B.
			%
			Yhid=f(A*Inp(:,k)); % implement passo forward (come riga 27) -> spezzato in 2 perchè ci servono i valori di uscita dei neuroni hidden su yhid sia quelli sui neuroni di output out
			Out=f(B*Yhid); % come sopra (continuo)
			DOut=(Targ(:,k)-Out).*Out.*(1-Out); % segnale d'errore delta target-output (derivata funz sigmoidale)
			E=DOut'*B; % correzione ai pesi utilizzando DOut (E contiene la sommatoria dei prodotti del segnale d'errore Dout per i pesi dei neuroni di uscita successivi)
            B=B+DOut*Yhid'; %update pesi di secondo livello (secondo E)
			DYhid=Eta*E'.*Yhid.*(1-Yhid); % segnale d'errore dei valori sui neuroni hidden (matrice E contiene segnale d'errore cumulativo che proviene dai neuroni successivi)
			A=A+DYhid*Inp(:,k)'; %update pesi di primo livello utilizzando DYhid
			% 
		end
	end
end
R=R';
