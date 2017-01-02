function [ID, CP, HP, stardust, level, cir_center] = pokemon_stats (img, model)
% Please DO NOT change the interface
% INPUT: image; model(a struct that contains your classification model, detector, template, etc.)
% OUTPUT: ID(pokemon id, 1-201); level(the position(x,y) of the white dot in the semi circle); cir_center(the position(x,y) of the center of the semi circle)

svmStruct = model.svmStruct;
codeBook =  model.codeBook;

cpTemplate = ((imread('assets/cp.bmp')));
hpTemplate = ((imread('assets/hp.bmp')));
hpTemplate2 = ((imread('assets/hp2.bmp')));
slsTemplate = ((imread('assets/sls.bmp')));
slsTemplate2 = ((imread('assets/sls2.bmp')));
powTemplate = ((imread('assets/pow.bmp')));
p00Template = ((imread('assets/p00.bmp')));
powUPTemplate = ((imread('assets/powUP.bmp')));


%{
% Replace these with your code
ID = 1;
CP = 123;
HP = 26;
stardust = 600;
level = [327,165];
cir_center = [355,457];
%}

[R, C , L] = size(img);
background = img;
if( L > 1)
background = (rgb2gray(img));
end;

ID = findID(background  , svmStruct  , codeBook, R , C);

[cir_center, level ] =  findLevel(background , R, C);

reImg = imresize(background , [1280 , 720]);

[R, C , L] = size(reImg);


CP = findCP( reImg , cpTemplate);
HP = findHP( reImg , hpTemplate , slsTemplate  , hpTemplate2 , slsTemplate2, R);
stardust = findPow( reImg , powTemplate  , p00Template , powUPTemplate ,R ,C);

end


function pId = findID(f , svmStruct  , codeBook , R , C )
    f = f(R / 8   : R / 2 - R / 15 , C/6 : C - C/6);
    feat = feature_extraction(f, codeBook);
    %f = imresize(f , [32 , 32]);
    %feat = extractHOGFeatures(f);
    predictedLabel = predict(svmStruct, feat);
    pId = predictedLabel;
end


function feat = feature_extraction(img , codeBook) 
    k = uint8(100); 
    feat = zeros(1 , k);
    [iR , iC , l] = size(img);
       
    surfPoints = detectSURFFeatures(img);
    [features, valid_points] = extractFeatures(img, surfPoints , 'Method' , 'SURF' , 'SURFSize' , 64);% my_sift(img);% 
    fsize = size(features);
    features = double(features);
    
        for i = 1 : size(features , 1)
            lfeat = features(i , :);
            euclDist = pdist2(lfeat , codeBook);
            [m , id] = min(euclDist);
            feat(1 , id) =  feat(1 , id) + 1;
        end;
    
end


function [o_cp] = findCP( background , template) 
    background = background(20:200 , :);
    background(background < max(max(background )) - 10) = 0;  %2 10 = 80
    [yoffSet, xoffSet] = fndPatch(template , background);

    sR = yoffSet+1 ;
    sC = xoffSet+1 + size(template,2);
    lr = size(template,2);
    lc = 34;
    [bR , bC] = size(background);
    cp = 0;
    if((sR + lr) < bR  && (sC + lc) < bC)
    temp = background(sR : sR + lr, sC : sC + lc  );
    [tr , tc] = size(temp);
    while( sC +( 3 * lc)  < bC &&  sum(sum(temp(:,5 :tc - 5  ))) > 0)
        count = 3;
        while(count > 0 && (sum(sum(temp(: , 1:2))) > 0 || sum(sum(temp(: , lc - 1:lc))) > 0))
            count = count - 1;
            sC = sC - 1;
            temp = background(sR : sR + lr, sC : sC + lc  );
        end;
      
        val = zeros(1, 10);
        for j = 0 : 9
            str = sprintf('assets/cp_%d.bmp' ,j);
            mask = imread(str);
            val(j + 1) = sum(sum(mask - temp));
            s = normxcorr2(mask , temp);
            val(j + 1)= max(s(:));
        end;
        [v , id] = max(val);
        %cp = [cp ,(id - 1)];
        cp = (cp * 10 ) + (id - 1);
        sC = sC + lc  ;
        temp = background(sR : sR + lr, sC : sC + lc  );
    
    end;
    end;
    
    o_cp = cp;
end

function [o_hp] = findHP( background  , hpTemplate , slsTemplate  , hpTemplate2 , slsTemplate2, R) 
    hR = uint16(R /2);
    background = background(hR - 70 :hR + 70 , :);
    [yoffSet1, xoffSet1] = fndPatch(hpTemplate , background);
    background(1 : yoffSet1 - 40 , :) = 0;
    background( yoffSet1 + 40   :  size(background , 1) , :) = 0;
    [yoffSet2, xoffSet2] = fndPatch(slsTemplate , background);

    hp = [];
    background = 255 - background;
    tresh = 70;
    background(background > tresh) = 255;
    background(background < tresh) = 0;
    cr = 1;
    if(xoffSet2 > xoffSet1)
        %hp frst
        sR = yoffSet1 ;
        sC = xoffSet1+2+ size(hpTemplate,2);
        lr = size(hpTemplate,1);
        lc = 5;
        hp = identifyNumber(background , xoffSet2, 3, sR, sC, lr , lc , 5 , 1 ,1 , 1);
    else
        [yoffSet1, xoffSet1] = fndPatch(hpTemplate2 , background);
        [yoffSet2, xoffSet2] = fndPatch(slsTemplate2 , background);
        sR = yoffSet1 ;
        sC = xoffSet1  ;
        lr = size(hpTemplate,1);
        lc = 2;
        hp = identifyNumber(background ,  xoffSet2 +  size(slsTemplate,2), 3, sR, sC, lr , lc , 5 , 1 ,2 , -1);
    end;
o_hp = hp;
end

function [o_pow] = findPow( background  , powTemplate  , p00Template , powUPTemplate ,R ,C) 

    [yoffSet, xoffSet] = fndPatch(powUPTemplate , background);
    background = background(yoffSet - 10 : yoffSet + size(powUPTemplate,1) + 10 , :  );
    [yoffSet1, xoffSet1] = fndPatch(powTemplate , background);
    [yoffSet2, xoffSet2] = fndPatch(p00Template , background);
    temp = background(yoffSet2 : yoffSet2 + size(p00Template,1) , xoffSet1 + size(powTemplate,2) : xoffSet2  );
    [tR , tC] = size(temp);
    pwRange = (xoffSet2 - (xoffSet1 + size(powTemplate,2)));
    pwRange = abs(pwRange -  size(powTemplate,2));
    step = 1;
    flag = 0 ;
    if(pwRange > 10)
        %two digts
        step =2;
        flag = 1;
    end;
    br = 10;
    or = temp;
    pow = 0;
    while(br < tC && step ~= 0)
        if(flag == 1)
            if (step == 2)
                temp2 = zeros(size(temp));
                temp2(temp <= 200)  = 1;%190
                
                while( sum(sum(temp2(: , br))) ~=0 && br < tC - 1)
                    br = br + 1;
                end;
                br = br + 1;
                temp = or(: , 1 : br);
            else
                temp = or(: , br : size(or  ,2 ));
            end;
        end;
        step = step -1;
     for j = 0 : 9
            str = sprintf('assets/po_%d.bmp' ,j);
            mask = imread(str);
            temp =  imresize(temp ,  size(mask));
            s = normxcorr2(mask , temp);
            val(j + 1)= max(s(:));
        end;
        [v , id] = max(val);
         %pow = [pow , id - 1];
         pow = (pow * 10) + (id - 1);
    end;
    %pow = [pow , 0 , 0];
    pow = pow * 100;
    o_pow = pow;
end

function [oCenter, oLevel] =  findLevel(forLevel , R , C)
    err = 18;
    Rmin = 200;
    Rmax = 700;
    orn = forLevel;
    lvltemplate = imread('assets/lvl.bmp');
    [center, radius] = imfindcircles(forLevel,[Rmin Rmax],'ObjectPolarity','bright' , 'Sensitivity',0.99);
    %viscircles(center, radius,'EdgeColor','b');
    h = imhist(forLevel);
    [v , d] = max(h);
    forLevel  = (forLevel >= (d - 2));
    forLevel(uint16(R/2) : R , :) = 0;
    forLevel(1 : 50, :) = 0;
    se = strel('disk',1);
    forLevel = imdilate(forLevel,se);
    flag = true;
    count = 1;
     %{%
     if(numel(center) >= 2)
          cC = ( center( : , 2 ));
          cR = ( center( : , 1 ));
          cC = abs(cC - C/2);
           ids =  find( cC < 200);
           tempC = [];
           tempR = [];
           for j = 1 : size(ids , 1)
               if(radius(ids(j)) <= C/2)
                   if(cR(ids(j)) < R/2 )
                    tempC(count, :) = center( ids(j), :) ;
                    tempR (count , :)= radius( ids(j), :);
                    count = count + 1;
                   end
               end;
           end;
            center = tempC;
            radius = tempR;
      end;
 
%}
 if(numel(radius) ~= 0)
     off = 40;
     cMask = zeros(R , C);
     [columnsInImage rowsInImage] = meshgrid(1:C, 1:R);
     cMask = ((rowsInImage - center(1 ,2)).^2  + (columnsInImage - center(1 ,1)).^2 <= (radius(1 ,1) + off).^2);
     cMask(center(1 ,2) + off : R , :) = 0;
     forLevel = forLevel & cMask;
     off = 70;
     cMask = ((rowsInImage - center(1 ,2)).^2  + (columnsInImage - center(1 ,1)).^2 >= (radius(1 ,1) - off).^2);
     off = 0;
     cMask(center(1 ,2) + off : R , :) = 0;
     forLevel = forLevel & cMask;
     [yoffSet, xoffSet] = fndPatch(lvltemplate , forLevel);

     oCenter = center(1 , :);
     oLevel = [uint16(yoffSet(1 ,1)  -  (size(lvltemplate,1) / 2) + err ), uint16(xoffSet(1 ,1) - (size(lvltemplate,2) /2) + err) ];
else
     %happens when there no crl or less power
     forLevel(1 :  R/6, :) = 0;
     forLevel(uint16(R/2 - R/7) : R , :) = 0; 
     se = strel('disk',4);
     forLevel = imopen(forLevel,se);
    [yoffSet, xoffSet] = fndPatch(lvltemplate , forLevel);
    oLevel = [uint16(yoffSet(1 ,1)  -  (size(lvltemplate,1) / 2) + err), uint16(xoffSet(1 ,1) - (size(lvltemplate,2) /2) + err) ];
    oCenter = [ uint16(C/2) ,   oLevel(1 ,1)];
 end;
 x = oLevel(1 , 1);
 oLevel(1 , 1) = oLevel(1 , 2);
 oLevel(1 , 2) = x;
 
 %{
 x = oCenter(1,1);
 oCenter(1 , 1) = oCenter(1 , 2);
 oCenter(1 , 2) = x;
 %}
end

function num_O = identifyNumber(background , patchLen, patchLenErr, sR, sC, lr , lc , lcErr , scErr,tempType , dir)
    num = [];   
    num_O = 0;
     [bR , bC] = size(background);
    if((sR + lr) < bR  && (sC + lc) < bC && (sC - lc) > 1)
    while((sC + lc) < bC && sum(sum(background(sR : sR + lr,  sC  ))) == 0)
        sC = sC + dir;
    end;
    sC = sC - (scErr * dir);
    if(dir == 1)
        flag = sC <= patchLen - patchLenErr;
    else
        flag = sC >= patchLen + patchLenErr;
    end;
   while(flag)
        while( sC + (lc * dir) > 1 &&  sC + (lc * dir) < bC && sum(sum(background(sR : sR + lr,  sC + (lc * dir)  ))) ~=0)
              lc = lc + 1;  
        end;
         if(dir == 1)
            temp = background(sR : sR + lr, sC : sC + lc  );
        else
            temp = background(sR : sR + lr,  sC - lc : sC  );
        end;
        if(sum(sum(temp)) == 0)
            break;
        end;
        for j = 0 : 9
            str = [];
            if(tempType == 1)
                str = sprintf('assets/hp1_%d.bmp' ,j); %pre read later
            elseif(tempType == 2)
                str = sprintf('assets/hp2_%d.bmp' ,j);
            end;
            mask = imread(str);
            temp =  imresize(temp ,  size(mask));
            s = normxcorr2(mask , temp);
            val(j + 1)= max(s(:));
        end;
        [v , id] = max(val);
        if(dir == 1)
           num = [num ,(id - 1)];
        else
           num = [(id - 1) ,num ]; 
        end;
        sC = sC + (lc * dir);
        lc = lcErr;
        if(dir == 1)
            flag = sC <= patchLen - patchLenErr;
        else
            flag = sC >= patchLen + patchLenErr;
        end;
   end
   for j = 1 : size(num , 2)
       num_O = ( num_O * 10 ) + num(1 , j);
   end;
   end;
end

function [yoffSet, xoffSet] = fndPatch(template , background)
    c = normxcorr2(template , background );
    [ypeak, xpeak] = find(c==max(c(:)));
    yoffSet = ypeak-size(template,1);
    xoffSet = xpeak-size(template ,2);
    
    [br , bc] = size(background);
   
    
    if(size(yoffSet , 1) > 1)
        yoffSet = yoffSet(1,1);
    end;
    
    if(size(xoffSet , 1) > 1)
        xoffSet = xoffSet(1,1);
    end;
    
     if(yoffSet <= 1)
        yoffSet = uint16(br/ 2);
    end;
    
    if(xoffSet <= 1)
        xoffSet = uint16(bc/ 2);
    end;
    
    
     if(yoffSet >= br)
        yoffSet = uint16(br/ 2);
    end;
    
    if(xoffSet >= bc)
        xoffSet = uint16(bc/ 2);
    end;
    
end

