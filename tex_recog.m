function [maskT,L]=tex_recog(I0)
[X,pixelIdxList,N]=image_superpixel(I0);
kN=clustersCal(X);
idx = spectralcluster(double(X),kN,'Distance',@FAdist);
[m,n,~]=size(I0);
L=zeros(m,n);
for i=1:N
    L(pixelIdxList{i})=idx(i);
end
maskT= labeloverlay(I0,L);
figure,imshow(I0)
end
%%
%生成超像素关联矩阵
    function [X,pixelIdxList,N]=image_superpixel(I0)
        [m,n,~]=size(I0);
        gausFilter=fspecial('gaussian',[5 5],2.6);I=imfilter(I0,gausFilter);%
        Alab = single(rgb2lab(I));
    
        Lsp = single(computeLBP(rgb2gray(I0)));
        Med = single(entropyfilt(rgb2gray(I0)));
        G=single(calDI(rgb2gray(I0)));
        [TSP, TSOP] = computeTSP_TSOP_vectorized(G, Lsp, Med);
         %visualize_results(I,G,Lsp,Med,TSP, TSOP)
        %超像素数目尺度设定
        if m*n>=120000
            M=100;
        else
            M=25;
        end
        
        [LL,N] = superpixels(Alab,round(m*n/M),'isInputLab',true);
        pixelIdxList = label2idx(LL);
        A1 = zeros(N,1,'single');
        A2 = zeros(N,1,'single');
        A3 = zeros(N,1,'single');
        A4 = zeros(N,1,'single');
        A5 = zeros(N,1,'single');
        for i = 1:N
            A1(i)=mean(Alab(pixelIdxList{i}));
            A2(i)=mean(Alab(pixelIdxList{i}+m*n));
            A3(i)=mean(Alab(pixelIdxList{i}+2*m*n));
            A4(i) = mean(TSP(pixelIdxList{i}));
            A5(i) = mean(TSOP(pixelIdxList{i}));
        end
        X=[A1,A2,A3,A4,A5];
        X=(X-min(X))./(max(X)-min(X));
    end
%%
    function [TSP, TSOP] = computeTSP_TSOP_vectorized(G, Lbp, Med)
        A = G + Lbp;
        phi = (pi/2) * Med;
        TSP = A;  
        TSOP = A .* cos(phi);  
    end

    function kN = clustersCal(A)
    A = im2double(A);   
    Z=dist(A','cosine');
    Z = (Z + Z') / 2;

    % 验证矩阵性质
    if ~issymmetric(Z)
        warning('矩阵非对称，进行对称化处理');
        Z = (Z + Z') / 2;
    end

        C = eig(Z);

        if ~isreal(C)
            C = real(C);
        end
        C = round((C-min(C))/(max(C)-min(C))*10);
        C=unique((C));
        gaps = diff(C);
        kN = sum(gaps >=1)+1;  
end
    function lbp = computeLBP(image) 
        P = 8; 
        R = 1; 
        [rows, cols] = size(image);
        lbp = zeros(rows, cols);
        neighbor_offsets = [
            -1, -1;  
            -1,  0; 
            -1,  1;  
            0,  1;  
            1,  1;  
            1,  0;  
            1, -1;  
            0, -1   
            ];
        for i = (1+R):(rows-R)
            for j = (1+R):(cols-R)
                center = double(image(i, j));
                lbp_val = 0;
                for k = 1:P
                    x = i + neighbor_offsets(k, 1);
                    y = j + neighbor_offsets(k, 2);
                    neighbor = double(image(x, y));
                    if neighbor >= center
                        lbp_val = lbp_val + 2^(k-1);
                    end
                end
                lbp(i, j) = lbp_val;
            end
        end
    end

    function D = FAdist(ZI, ZJ)
        % FAdist 融合马氏距离和余弦距离的自适应距离
        if isempty(ZJ)
            D = [];
            return;
        end
        delta = ZI - ZJ;
        try
            S = cov(ZJ);
            S_reg = S + 1e-8 * eye(size(S));
            S_inv = S_reg \ eye(size(S)); 

            mahalanobis_sq = sum((delta * S_inv) .* delta, 2);
            D2 = sqrt(max(mahalanobis_sq, 0));
        catch
            D2 = sqrt(sum(delta.^2, 2)); 
        end

        % 计算余弦距离
        norm_products = sqrt(sum(ZI.^2)) * sqrt(sum(ZJ.^2, 2));
        dot_products = ZJ * ZI';
        valid_norms = norm_products > eps;
        D1 = ones(size(dot_products));
        D1(valid_norms) = 1 - dot_products(valid_norms) ./ norm_products(valid_norms);
        sigmoid_val = 1 ./ (1 + exp(-2* (D2 - 0.5)));

        D = D2 .* sigmoid_val + D1 .* (1 - sigmoid_val);
        D = max(D, 0);
        D(isnan(D) | isinf(D)) = realmax('double');
    end


    function DI=calDI(I)
        if ~isa(I,'double')
            I = double(I)/255;   
        end
        [~,~,nchannel]=size(I) ;
        if nchannel==1
            DI=im2uint8(calDF(I));
        elseif nchannel==3
            DI(:,:,1)=im2uint8(calDF(I(:,:,1)));
            DI(:,:,2)=im2uint8(calDF(I(:,:,2)));
            DI(:,:,3)=im2uint8(calDF(I(:,:,3)));
        end
        function DF=calDF(im)
            proci=padarray(im,[1 1],'replicate');
            [row,clm]=size(proci);
            Dx=(proci(2:row-1,3:clm)-proci(2:row-1,1:clm-2))/2;
            Dy=(proci(3:row,2:clm-1)-proci(1:row-2,2:clm-1))/2;
            DF=sqrt(Dx.^2+Dy.^2);
        end
    end

    function visualize_results(I,G,Lsp,Med,TSP, TSOP)
% 可视化所有结果
    figure('Position', [100, 100, 1400, 900]);
    % 原始图像
    subplot(2, 3, 1);
    imshow(I); title('原始图像');   
    % 超像素分割
    subplot(2, 3, 2);  
    imagesc(G);
    title('Gradient');
    axis image;%  colorbar;
    % colormap(jet);
     % 纹理结构势能 (Lsp)
    subplot(2, 3, 3);
    imagesc(Lsp); 
   axis image; 
   title('Lsp');
    % colormap(jet);
    % 纹理结构势能 (Lsp)
    subplot(2, 3, 4);
    imagesc(Med); 
    axis image; 
   title('Med');
    % colormap(jet);
    % 纹理结构势能 (TSP)
    subplot(2, 3, 5);
    imagesc(TSP); 
    axis image; 
    %colorbar;
    title('TSP');
    % colormap(jet); 
    % 纹理结构序参量 (TSOP)
    subplot(2, 3, 6);
    imagesc(TSOP); axis image; % colorbar;
    title('TSOP');
    colormap(jet);
   
    end

 
    
