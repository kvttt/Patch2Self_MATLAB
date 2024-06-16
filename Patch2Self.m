function [denoised_arr] = Patch2Self( ...
    data, bvals, patch_radius, model, b0_threshold, alpha, b0_denoising ...
)
% PATCH2SELF (MATLAB reimplementation)
%
% References:
% [Fadnavis20] S. Fadnavis, J. Batson, E. Garyfallidis, Patch2Self:
%              Denoising Diffusion MRI with Self-supervised Learning,
%              Advances in Neural Information Processing Systems 33 (2020)
%
% Adopted from:
% https://github.com/dipy/dipy/blob/master/dipy/denoise/patch2self.py
% 
% Kaibo, 2024

    % segregates volumes by b0 threshold
    b0_idx = bvals <= b0_threshold;
    dwi_idx = bvals > b0_threshold;

    data_b0s = squeeze(data(:,:,:,b0_idx));
    data_dwis = squeeze(data(:,:,:,dwi_idx));

    % create empty arrays
    denoised_b0s = zeros(size(data_b0s));
    denoised_dwis = zeros(size(data_dwis));

    denoised_arr = zeros(size(data));

    % if only 1 b0 volume, skip denoising it
    if ndims(data_b0s) == 3 || b0_denoising == false
        denoised_b0s = data_b0s;
    else
        train_b0 = extract_3d_patches( ...
            padarray(data_b0s, [2 * patch_radius, 0], 0, "both"), ...
            patch_radius ...
        );
        
        for i = 1:size(data_b0s, 4)
            denoised_b0s(:,:,:,i) = vol_denoise( ...
                train_b0, i, model, size(data_b0s), alpha ...
            );
        end
    end

    % Separate denoising for DWI volumes
    train_dwis = extract_3d_patches( ...
        padarray(data_dwis, [2 * patch_radius, 0], 0, "both"), ...
        patch_radius ...
    );

    for i = 1:size(data_dwis, 4)
        denoised_dwis(:,:,:,i) = vol_denoise( ...
            train_dwis, i, model, size(data_dwis), alpha ...
        );
    end

    denoised_arr(:,:,:,b0_idx) = denoised_b0s;
    denoised_arr(:,:,:,dwi_idx) = denoised_dwis;

end


function [all_patches] = extract_3d_patches(arr, patch_radius)

    patch_size = 2 * patch_radius + 1;
    dim = size(arr, 4);
    all_patches = zeros( ...
        (size(arr, 1) - 2 * patch_radius(1)) * ...
        (size(arr, 2) - 2 * patch_radius(2)) * ...
        (size(arr, 3) - 2 * patch_radius(3)), ...
        prod(patch_size, "all"), ...
        dim ...
    );

    idx = 1;
    for i = patch_radius(1) + 1:size(arr, 1) - patch_radius(1)
        for j = patch_radius(2) + 1:size(arr, 2) - patch_radius(2)
            for k = patch_radius(3) + 1:size(arr, 3) - patch_radius(3)
                ix1 = i - patch_radius(1);
                ix2 = i + patch_radius(1);
                jx1 = j - patch_radius(2);
                jx2 = j + patch_radius(2);
                kx1 = k - patch_radius(3);
                kx2 = k + patch_radius(3);
                X = reshape( ...
                    arr(ix1:ix2, jx1:jx2, kx1:kx2, :), ...
                    [prod(patch_size, "all"), dim] ...
                );
                all_patches(idx, :, :) = X;
                idx = idx + 1;
            end
        end
    end

    all_patches = permute(all_patches, [3, 2, 1]);

end


function [pred] = vol_denoise(train, vol_idx, model, data_shape, alpha)

    [X, y] = vol_split(train, vol_idx);

    if model == "ols"
        beta = (X.' * X) \ (X.' * y);
    elseif model == "ridge"
        beta = (X.' * X + alpha * eye(size(X, 2))) \ (X.' * y);
    end

    pred = reshape(X * beta, [data_shape(1), data_shape(2), data_shape(3)]);

end


function [cur_x, y] = vol_split(train, vol_idx)

    mask = zeros(size(train,1), 1);
    mask(vol_idx) = 1;
    cur_x = train(mask == 0, :, :);
    cur_x = reshape( ...
        cur_x, ...
        [(size(train, 1) - 1) * size(train, 2), size(train, 3)] ...
    );
    cur_x = permute(cur_x, [2, 1]);

    y = squeeze(train(vol_idx, (size(train, 2) + 1) / 2, :));

end
