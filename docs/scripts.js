document.addEventListener('DOMContentLoaded', () => {
    // --- 設定 ---
    const MAX_ITERATION_VALUE = 100; // mask_iter_100.png が最大
    const ITERATION_STEP = 10;       // イテレーションは10刻み
    const NUM_SLIDER_STEPS = MAX_ITERATION_VALUE / ITERATION_STEP + 1; // スライダーのステップ数 (0, 10, ..., 100 なので11ステップ)

    const IMAGE_BASE_PATH = 'images/'; // HTMLファイルからの相対パス
    const FULL_SIZE_DIR = 'full_size/';
    const CLIP_DIRS = {
        'cliped_x160y120r60': 'cliped_x160y120r60/',
        'cliped_x175y210r90': 'cliped_x175y210r90/'
    };
    const IMAGE_NAMES = {
        original: 'image.png',
        direction: 'direction.png',
        maskPrefix: 'mask_iter_'
    };

    // --- DOM要素の取得 ---
    const clipSelect = document.getElementById('clip-select');
    const iterationSlider = document.getElementById('iteration-slider');
    
    const fullOriginalImage = document.getElementById('full-original-image');
    const fullDirectionImage = document.getElementById('full-direction-image');
    const fullMaskImage = document.getElementById('full-mask-image');
    const fullIterLabel = document.getElementById('full-iter-label');

    const clippedOriginalImage = document.getElementById('clipped-original-image');
    const clippedDirectionImage = document.getElementById('clipped-direction-image');
    const clippedMaskImage = document.getElementById('clipped-mask-image');
    const clippedIterLabel = document.getElementById('clipped-iter-label');

    const toggleDirectionCheckbox = document.getElementById('toggle-direction-image');
    const fullDirectionSet = document.getElementById('full-direction-set');
    const clippedDirectionSet = document.getElementById('clipped-direction-set');

    // --- 初期設定 ---
    iterationSlider.min = 0;
    iterationSlider.max = NUM_SLIDER_STEPS - 1; // スライダーの値は 0, 1, 2, ..., 10
    iterationSlider.step = 1;
    iterationSlider.value = 0; // 初期値

    // --- 表示切り替え関数 ---
    function toggleDirectionImageVisibility() {
        const isVisible = toggleDirectionCheckbox.checked;
        if (fullDirectionSet) {
            fullDirectionSet.style.display = isVisible ? 'block' : 'none';
        }
        if (clippedDirectionSet) {
            clippedDirectionSet.style.display = isVisible ? 'block' : 'none';
        }
    }


    // --- 画像更新関数 ---
    function updateImages() {
        const selectedClipKey = clipSelect.value;
        // スライダーの値 (0, 1, ..., 10) を実際のイテレーション番号 (0, 10, ..., 100) に変換
        const sliderStepValue = parseInt(iterationSlider.value, 10);
        const iteration = sliderStepValue * ITERATION_STEP;

        const currentClipDir = CLIP_DIRS[selectedClipKey];
        if (!currentClipDir) {
            console.error('Invalid clip directory key:', selectedClipKey);
            return;
        }

        // フルサイズ画像のパス設定
        fullOriginalImage.src = `${IMAGE_BASE_PATH}${FULL_SIZE_DIR}${IMAGE_NAMES.original}`;
        fullDirectionImage.src = `${IMAGE_BASE_PATH}${FULL_SIZE_DIR}${IMAGE_NAMES.direction}`;
        fullMaskImage.src = `${IMAGE_BASE_PATH}${FULL_SIZE_DIR}${IMAGE_NAMES.maskPrefix}${iteration}.png`;
        fullIterLabel.textContent = iteration;

        // クリップ画像のパス設定
        clippedOriginalImage.src = `${IMAGE_BASE_PATH}${currentClipDir}${IMAGE_NAMES.original}`;
        clippedDirectionImage.src = `${IMAGE_BASE_PATH}${currentClipDir}${IMAGE_NAMES.direction}`;
        clippedMaskImage.src = `${IMAGE_BASE_PATH}${currentClipDir}${IMAGE_NAMES.maskPrefix}${iteration}.png`;
        clippedIterLabel.textContent = iteration;

        // スライダーラベル更新処理を削除

        // エラーハンドリング（オプション）
        [fullOriginalImage, fullDirectionImage, fullMaskImage, 
         clippedOriginalImage, clippedDirectionImage, clippedMaskImage].forEach(img => {
            img.onerror = () => {
                console.warn(`Failed to load image: ${img.src}`);
                // img.alt = `Error loading: ${img.src.split('/').pop()}`; // 代替テキスト表示
            };
        });
    }

    // --- イベントリスナー ---
    clipSelect.addEventListener('change', updateImages);
    iterationSlider.addEventListener('input', updateImages);
    toggleDirectionCheckbox.addEventListener('change', toggleDirectionImageVisibility);

    // --- 初期画像読み込み & 表示設定 ---
    // updateImagesを呼び出す前にスライダーの初期値を反映させる
    updateImages();
    toggleDirectionImageVisibility(); // 初期表示状態をチェックボックスに合わせる
});
