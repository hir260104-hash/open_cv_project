#define _CRT_SECURE_NO_WARNINGS
#include "opencv2/opencv.hpp"
#include <iostream>
#include <ctime>
#include <vector>
#include <string>
#include <sstream>
#include <thread>

#pragma comment(lib, "winmm.lib") // MCI 함수 사용을 위한 라이브러리 링크
#include <Windows.h> // Beep 소리를 위해 추가
#include <commdlg.h>
#define IS_WINDOWS true
#pragma comment(lib, "comdlg32.lib")

// 효과 종류 정의
enum EffectType {
	INVERSE,
	AFFINE_COMBO, // 종합 어파인 변환
	PERSPECTIVE,  // 투시 변환
	FLIP,
	RESIZE_ACTION,
	MORPH,
	EYE_EMPTY,
	MOUTH_EMPTY,
	FACE_EMPTY,
	EMBOSSING,
	SHARPENING,
	NOISE,
	INVERT,
	GRAYSCALE,
	SEPIA,
	COLOR_SWAP, // BGR -> RGB
	COUNT
};

// 선택 모드를 관리하기 위한 전역 변수
enum SelectionMode { NONE, FILE_MODE, CAM_MODE };
SelectionMode g_mode = NONE;

// 초기화 함수: 게임 시작 시 한 번만 호출
void initGameSounds() {
	// 미리 파일을 열고 별칭(alias)을 부여함
	mciSendStringA("open \"pop.mp3\" type mpegvideo alias hit_sound", NULL, 0, NULL);
	mciSendStringA("open \"claps.mp3\" type mpegvideo alias claps_sound", NULL, 0, NULL);
}

void playHitSound() {
	mciSendStringA("play hit_sound from 0", NULL, 0, NULL);
}

void playClapsSound() {
	mciSendStringA("play claps_sound from 0", NULL, 0, NULL);
}

// 랜덤 위치 반환
cv::Point getRandomPosition(int width, int height, int radius) {
	if (width <= 2 * radius || height <= 2 * radius) return cv::Point(width / 2, height / 2);
	int x = rand() % (width - 2 * radius) + radius;
	int y = rand() % (height - 2 * radius) + radius;
	return cv::Point(x, y);
}


static int effectIndex = 0;
int g_currentEffectIdx = -1; // 현재 무슨 효과인지 저장할 전역 변수

int spawnTimer = 0; // 카운터
int spawnDelay = 60; // [조절] 30~60 정도면 적당함 (숫자가 클수록 늦게 나타남)

class FaceBall {
public:
	cv::Point position; // x, y
	int radius; // 공의 반지름
	int displayRadius; // 애니메이션을 위한 현재 표시 반지름
	cv::Mat originalFace; // 불러온 얼굴 이미지
	cv::Mat currentFace; // 효과가 적용된 현재 얼굴 이미지
	cv::Mat originalMask; // 원본 정원 마스크 보관용
	cv::Mat mask;         // 변형된 마스크가 저장될 곳
	int hitCount = 0; // 때릴수록 빨개짐 (누적)
	int effectTimer = 0;
	bool active = true; // 터치 감지를 위해 추가

	FaceBall(std::string path, int r) { // 생성자, Ball의 객체
		radius = r;
		displayRadius = r;
		originalFace = cv::imread(path);

		if (originalFace.empty()) {
			// 이미지 없으면 빨간 원으로 대체
			originalFace = cv::Mat::zeros(cv::Size(r * 2, r * 2), CV_8UC3);
			cv::circle(originalFace, cv::Point(r, r), r, cv::Scalar(0, 0, 255), -1);
		}
		resize(originalFace, originalFace, cv::Size(r * 2, r * 2));

		// 원본 마스크 딱 한 번만 생성해서 저장
		originalMask = cv::Mat::zeros(originalFace.size(), CV_8UC1);
		cv::circle(originalMask, cv::Point(r, r), r, cv::Scalar(255), -1);

		currentFace = originalFace.clone();
		mask = originalMask.clone();
	}

	FaceBall(cv::Mat capturedImg, int r) {
		radius = r;
		displayRadius = r;
		if (capturedImg.empty()) {
			originalFace = cv::Mat::zeros(cv::Size(r * 2, r * 2), CV_8UC3);
			cv::circle(originalFace, cv::Point(r, r), r, cv::Scalar(0, 0, 255), -1);
		}
		else {
			cv::resize(capturedImg, originalFace, cv::Size(r * 2, r * 2));
		}
		currentFace = originalFace.clone();

		// [수정] 원본 마스크를 흰색 정사각형으로 생성
		originalMask = cv::Mat::zeros(originalFace.size(), CV_8UC1);

		// (0, 0)부터 (2r, 2r)까지 꽉 채운 사각형 그리기
		cv::rectangle(originalMask, cv::Point(0, 0), cv::Point(r * 2, r * 2), cv::Scalar(255), -1);

		currentFace = originalFace.clone();
		mask = originalMask.clone();
	}

	void applyRandomEffect(int width, int height) {
		hitCount++; // 때릴 때마다 누적

		playHitSound();

		g_currentEffectIdx = effectIndex;

		int totalEffects = COUNT; // 총 효과 개수
		
		// [핵심] 변환 전에 항상 원본 상태를 복사해서 새로 시작 (누적 방지)
		currentFace = originalFace.clone();
		mask = originalMask.clone();

		// 2. [해결] 사라진 빨간색 효과 다시 적용
		// 누적된 hitCount에 따라 빨간색(BGR 중 R채널)을 더해줌
		currentFace += cv::Scalar(0, 0, min(hitCount * 10, 255));

		effectTimer = 15; // 효과 유지 프레임 수
		displayRadius = radius * 1.5; // 터치 시 기본적으로 커짐 (크기변환)

		float wf = (float)originalFace.cols;
		float hf = (float)originalFace.rows;

		// 변환 시 잘림 방지를 위한 공통 설정
		// BORDER_REFLECT_101: 경계선을 기준으로 거울처럼 반사하여 빈 곳을 채움
		int borderMode = cv::BORDER_REFLECT_101;

		switch (effectIndex) {
		case EMBOSSING:
		{
			cv::Mat kernel = (cv::Mat_<float>(3, 3) << -1, -1, 0, -1, 0, 1, 0, 1, 1);
			cv::filter2D(currentFace, currentFace, -1, kernel, cv::Point(-1, -1), 128);
			break;
		}
		case SHARPENING:
		{
			cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
			cv::filter2D(currentFace, currentFace, -1, kernel);
			break;
		}
		case NOISE:
		{
			cv::Mat noise(currentFace.size(), currentFace.type());
			cv::randn(noise, 0, 100);
			currentFace += noise;
			break;
		}
		case INVERT:
			cv::bitwise_not(currentFace, currentFace);
			break;
		case GRAYSCALE:
		{
			cv::Mat gray;
			cv::cvtColor(currentFace, gray, cv::COLOR_BGR2GRAY);
			cv::cvtColor(gray, currentFace, cv::COLOR_GRAY2BGR);
			break;
		}
		case SEPIA:
		{
			cv::Mat m = (cv::Mat_<float>(3, 3) <<
				0.131, 0.534, 0.272,
				0.168, 0.686, 0.349,
				0.189, 0.769, 0.393);
			cv::transform(currentFace, currentFace, m);
			break;
		}
		case COLOR_SWAP:
			cv::cvtColor(currentFace, currentFace, cv::COLOR_BGR2RGB);
			break;
		case INVERSE:
			// 원본 얼굴의 색상을 완전히 뒤집기
			currentFace = ~originalFace;
			break;
		case FLIP: // 반전 및 대칭
		{
			cv::flip(currentFace, currentFace, -1);
			cv::flip(mask, mask, -1);
			break;
		}
		case AFFINE_COMBO:
		{
			cv::Point2f center(wf / 2.0f, hf / 2.0f);
			double angle = (rand() % 61) - 30;

			// [중요] 정사각형 대각선 길이는 한 변의 약 1.414배야.
			// 찌그러짐(Shear)까지 고려해서 스케일을 0.5 정도로 낮춰야 절대 안 잘림.
			double scale = 0.5 + (rand() % 11) / 100.0;

			cv::Mat M = cv::getRotationMatrix2D(center, angle, scale);

			// 축 이동 범위를 좁혀서 경계선 탈출 방지
			M.at<double>(0, 2) += (rand() % 7 - 3);
			M.at<double>(1, 2) += (rand() % 7 - 3);

			// 찌그러짐 (Shear) 적용
			M.at<double>(0, 1) += ((rand() % 21) - 10) / 100.0;
			M.at<double>(1, 0) += ((rand() % 21) - 10) / 100.0;

			// 변환 적용 (배경은 검은색으로 채움)
			cv::warpAffine(currentFace, currentFace, M, currentFace.size(),
				cv::INTER_LINEAR);
			cv::warpAffine(mask, mask, M, mask.size(),
				cv::INTER_LINEAR);
			break;
		}
		case PERSPECTIVE:
		{
			cv::Point2f srcPts[4] = { {0.f, 0.f}, {wf - 1.f, 0.f}, {wf - 1.f, hf - 1.f}, {0.f, hf - 1.f} };
			cv::Point2f dstPts[4];

			// [중요] 투시 변환 시에도 전체적으로 안쪽으로 모아야 안 잘림 (전체 20% 마진 기본 탑재)
			float padW = wf * 0.2f;
			float padH = hf * 0.2f;
			dstPts[0] = { padW, padH }; dstPts[1] = { wf - padW, padH };
			dstPts[2] = { wf - padW, hf - padH }; dstPts[3] = { padW, hf - padH };

			int dir = rand() % 4;
			float m = wf * 0.25f; // 추가 왜곡 강도

			if (dir == 0) { dstPts[0].x += m; dstPts[1].x -= m; } // 상단 왜곡
			else if (dir == 1) { dstPts[3].x += m; dstPts[2].x -= m; } // 하단 왜곡
			else if (dir == 2) { dstPts[0].y += m; dstPts[3].y -= m; } // 좌측 왜곡
			else { dstPts[1].y += m; dstPts[2].y -= m; } // 우측 왜곡

			cv::Mat M_p = cv::getPerspectiveTransform(srcPts, dstPts);
			cv::warpPerspective(currentFace, currentFace, M_p, currentFace.size(),
				cv::INTER_LINEAR);
			cv::warpPerspective(mask, mask, M_p, mask.size(),
				cv::INTER_LINEAR);
			break;
		}
		case RESIZE_ACTION:
		{
			// 1. 원본 데이터 유효성 검사 (가장 중요)
			if (originalFace.empty() || originalFace.cols == 0 || originalFace.rows == 0) {
				std::cout << "Error: originalFace is empty!" << std::endl;
				break;
			}

			// 2. 새로운 반지름 계산 및 범위 제한
			float randomScale = (rand() % 121 + 60) / 100.0f; // 0.6 ~ 1.8배
			int newRadius = (int)(40 * randomScale);

			// 반지름이 너무 작거나 너무 크면 고정 (화면 이탈 방지)
			if (newRadius < 20) newRadius = 20;
			if (newRadius > 150) newRadius = 150;

			radius = newRadius;
			displayRadius = radius;

			// 3. 리사이즈 (에러 방지를 위해 임시 Mat 사용 후 대입)
			try {
				cv::Mat tempFace, tempMask;
				cv::resize(currentFace, tempFace, cv::Size(radius * 2, radius * 2));
				cv::resize(originalMask, tempMask, cv::Size(radius * 2, radius * 2));

				currentFace = tempFace;
				mask = tempMask;
			}
			catch (const cv::Exception& e) {
				std::cout << "Resize Error: " << e.what() << std::endl;
			}

			effectTimer = 0;
			break;
		}
		case MORPH:
		{
			// 모폴로지 팽창(Dilation)을 강하게 주면 눈코입이 커지면서 서로 뭉쳐짐
			cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));

			morphologyEx(originalFace, currentFace, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 5);

			//cv::addWeighted(currentFace, 0.9, originalFace, 0.1, 0, currentFace);
			break;
		}
		case EYE_EMPTY:
		{
			cv::Mat displayFace = currentFace.clone();
			cv::Mat maskSource = originalFace.clone();
			cv::Mat debugImg = maskSource.clone(); // 여기에 사각형을 그려서 확인할 거야
			cv::Mat ycrcb, gray;
			cv::cvtColor(maskSource, ycrcb, cv::COLOR_BGR2YCrCb);
			cv::cvtColor(maskSource, gray, cv::COLOR_BGR2GRAY);

			cv::CascadeClassifier eyeCascade("haarcascade_eye.xml");
			std::vector<cv::Rect> eyes;
			eyeCascade.detectMultiScale(gray, eyes, 1.1, 3, 0, cv::Size(10, 10));

			// 전체 이미지 크기의 빈 마스크 생성
			cv::Mat fullMask = cv::Mat::zeros(maskSource.size(), CV_8UC1);

			for (cv::Rect eye : eyes) {
				// [수정] 샘플링 영역: 눈썹을 피하기 위해 눈 사각형 상단 내부의 아주 얇은 줄만 조준
				// 1. 샘플링 (눈 바로 위 좁은 영역)
				cv::Rect skinROI(eye.x, eye.y, 3, 3);
				cv::Mat debugROI = maskSource(skinROI).clone();
				cv::resize(debugROI, debugROI, cv::Size(100, 50)); // 너무 작으니까 키워서 보기
				cv::rectangle(debugImg, skinROI, cv::Scalar(0, 0, 255), 1);
				// cv::Rect skinROI(eye.x + eye.width / 4, eye.y + 2, eye.width / 2, 2);
				if (skinROI.y < 0 || skinROI.y >= ycrcb.rows) continue;

				cv::Mat sample = ycrcb(skinROI);
				cv::Scalar avgSkin = cv::mean(maskSource(skinROI));
				cv::Vec3b skinColor(avgSkin[0], avgSkin[1], avgSkin[2]);

				// 현재 눈 영역에 대한 히스토그램 계산
				cv::Mat hist;
				int channels[] = { 1, 2 };
				int histSize[] = { 32, 32 };
				float range[] = { 0, 256 };
				const float* ranges[] = { range, range };
				cv::calcHist(&sample, 1, channels, cv::Mat(), hist, 2, histSize, ranges);
				cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX);

				// 현재 눈 사각형 영역 안에서만 역투영 수행 (정확도 향상)
				cv::Mat eyeBP;
				cv::Mat eyeROI = ycrcb(eye);
				cv::calcBackProject(&eyeROI, 1, channels, hist, eyeBP, ranges);

				cv::Mat eyeMask;
				cv::threshold(eyeBP, eyeMask, 60, 255, cv::THRESH_BINARY_INV);

				// 2. [강력한 모폴로지] 확실하게 메우고 확장하기
// 커널 크기를 7x7 정도로 크게 잡아서 눈썹 영역까지 침범하게 만듦
				cv::Mat morphKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));

				// 소소한 구멍들을 메우고 (CLOSE)
				cv::morphologyEx(eyeMask, eyeMask, cv::MORPH_CLOSE, morphKernel, cv::Point(-1, -1), 3);
				// 흰색 영역을 사방으로 확 키워서 눈썹과 주변 그림자를 포함시킴 (DILATE)
				cv::morphologyEx(eyeMask, eyeMask, cv::MORPH_DILATE, morphKernel, cv::Point(-1, -1), 2);

				// 해당 눈 영역의 결과만 fullMask에 업데이트
				eyeMask.copyTo(fullMask(eye));

				// [합성 로직]
				int p = 15; // 패딩 복구 (경계선 제거 필수)
				cv::Rect wideEye;
				wideEye.x = max(0, eye.x - p);
				wideEye.y = max(0, eye.y - p);
				wideEye.width = min(displayFace.cols - wideEye.x, eye.width + 2 * p);
				wideEye.height = min(displayFace.rows - wideEye.y, eye.height + 2 * p);

				// [루프 내부]
				// 1. 현재 눈 주변의 '실시간' 평균 색상 추출 (빨개진 상태 포함)
				cv::Scalar currentAvg = cv::mean(displayFace(wideEye));
				cv::Vec3b dynamicSkinColor((uchar)currentAvg[0], (uchar)currentAvg[1], (uchar)currentAvg[2]);

				cv::Mat patch = displayFace(wideEye).clone();
				cv::Mat localMask = fullMask(wideEye);

				// patch에 색을 칠하기 전에 mask 자체를 부드럽게 깎음
				cv::Mat softMask;
				cv::GaussianBlur(localMask, softMask, cv::Size(21, 21), 0);
				softMask.convertTo(softMask, CV_32F, 1.0 / 255.0); // 0~1 사이 값으로 변환

				// 2. 패치에 색 입히기 (비율 조절)
				for (int i = 0; i < patch.rows; i++) {
					for (int j = 0; j < patch.cols; j++) {
						if (localMask.at<uchar>(i, j) == 255) {
							cv::Vec3b& p = patch.at<cv::Vec3b>(i, j);

							// 반영 비율 조절: 0.8이면 dynamicSkinColor(빨간 피부)를 80% 반영
							float blendRatio = 0.95f;
							p[0] = p[0] * (1.0f - blendRatio) + dynamicSkinColor[0] * blendRatio;
							p[1] = p[1] * (1.0f - blendRatio) + dynamicSkinColor[1] * blendRatio;
							p[2] = p[2] * (1.0f - blendRatio) + dynamicSkinColor[2] * blendRatio;
						}
					}
				}
				// 3. [패치 블러] (강하게 유지)
				cv::GaussianBlur(patch, patch, cv::Size(51, 51), 0);

				// 4. [중요] 최종 합성 방식 변경 (if문을 지우고 softMask 가중치 곱하기)
				for (int i = 0; i < patch.rows; i++) {
					for (int j = 0; j < patch.cols; j++) {
						float weight = softMask.at<float>(i, j);

						// 가중치를 약간 강화 (눈 중심은 확실히 지우기)
						weight = min(1.0f, weight * 1.5f);

						cv::Vec3b& target = displayFace.at<cv::Vec3b>(wideEye.y + i, wideEye.x + j);
						cv::Vec3b blurP = patch.at<cv::Vec3b>(i, j);

						for (int c = 0; c < 3; c++) {
							// weight가 높을수록(눈 중심) blurP(칠한 색), 낮을수록 target(원본)
							target[c] = (uchar)(blurP[c] * weight + target[c] * (1.0f - weight));
						}
					}
				}
			}
			displayFace.copyTo(currentFace);
			break;
		}
		case MOUTH_EMPTY:
		{
			cv::Mat displayFace = currentFace.clone();
			cv::Mat maskSource = originalFace.clone();

			cv::Mat debugImg = maskSource.clone(); // 여기에 사각형을 그려서 확인할 거야
			cv::Mat ycrcb, gray;
			cv::cvtColor(maskSource, ycrcb, cv::COLOR_BGR2YCrCb);
			cv::cvtColor(maskSource, gray, cv::COLOR_BGR2GRAY);

			// 1. 입 검출기 (경로 확인 필수)
			cv::CascadeClassifier mouthCascade("haarcascade_mcs_mouth.xml");
			if (mouthCascade.empty()) {
				std::cout << "모델 로드 실패! 경로 확인." << std::endl;
			}
			std::vector<cv::Rect> mouths;
			// 입은 보통 얼굴 하단에 있으므로 검출 범위를 제한하면 더 정확함
			mouthCascade.detectMultiScale(gray, mouths, 1.3, 5, 0, cv::Size(20, 20));

			cv::Mat fullMask = cv::Mat::zeros(maskSource.size(), CV_8UC1);

			for (cv::Rect mouth : mouths) {
				cv::rectangle(debugImg, mouth, cv::Scalar(0, 255, 255), 3);
				// [입 전용 샘플링] 입 바로 위(인중 근처) 피부색 추출
				cv::Rect skinROI(mouth.x + mouth.width / 4, max(0, mouth.y - 10), mouth.width / 2, 5);
				if (skinROI.y < 0 || skinROI.y >= ycrcb.rows) continue;

				

				cv::Mat sample = ycrcb(skinROI);
				cv::Mat hist;
				int channels[] = { 1, 2 };
				int histSize[] = { 32, 32 };
				float range[] = { 0, 256 };
				const float* ranges[] = { range, range };
				cv::calcHist(&sample, 1, channels, cv::Mat(), hist, 2, histSize, ranges);
				cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX);

				// 입 영역 역투영
				cv::Mat mouthROI = ycrcb(mouth);
				cv::Mat mouthBP;
				cv::calcBackProject(&mouthROI, 1, channels, hist, mouthBP, ranges);

				cv::Mat mouthMask;
				cv::threshold(mouthBP, mouthMask, 60, 255, cv::THRESH_BINARY_INV);

				// [모폴로지] 입은 면적이 넓으므로 커널을 더 크게
				cv::Mat morphKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
				cv::morphologyEx(mouthMask, mouthMask, cv::MORPH_CLOSE, morphKernel, cv::Point(-1, -1), 2);
				cv::dilate(mouthMask, mouthMask, morphKernel, cv::Point(-1, -1), 2);

				mouthMask.copyTo(fullMask(mouth));

				// [합성 로직] 입은 패딩을 더 넉넉하게 (p=30)
				int p = 25;
				cv::Rect wideMouth;
				wideMouth.x = max(0, mouth.x - p);
				wideMouth.y = max(0, mouth.y - p);
				wideMouth.width = min(displayFace.cols - wideMouth.x, mouth.width + 2 * p);
				wideMouth.height = min(displayFace.rows - wideMouth.y, mouth.height + 2 * p);

				cv::Scalar currentAvg = cv::mean(displayFace(wideMouth));
				cv::Vec3b dynamicSkinColor((uchar)currentAvg[0], (uchar)currentAvg[1], (uchar)currentAvg[2]);

				cv::Mat patch = displayFace(wideMouth).clone();
				cv::Mat localMask = fullMask(wideMouth);

				cv::Mat softMask;
				cv::GaussianBlur(localMask, softMask, cv::Size(51, 51), 0);
				softMask.convertTo(softMask, CV_32F, 1.0 / 255.0);

				for (int i = 0; i < patch.rows; i++) {
					for (int j = 0; j < patch.cols; j++) {
						if (localMask.at<uchar>(i, j) == 255) {
							cv::Vec3b& px = patch.at<cv::Vec3b>(i, j);
							float blendRatio = 0.95f; // 입술색은 강하므로 95% 이상 덮어야 함
							px[0] = px[0] * (1.0f - blendRatio) + dynamicSkinColor[0] * blendRatio;
							px[1] = px[1] * (1.0f - blendRatio) + dynamicSkinColor[1] * blendRatio;
							px[2] = px[2] * (1.0f - blendRatio) + dynamicSkinColor[2] * blendRatio;
						}
					}
				}
				cv::GaussianBlur(patch, patch, cv::Size(71, 71), 0); // 입은 더 뭉개야 함

				for (int i = 0; i < patch.rows; i++) {
					for (int j = 0; j < patch.cols; j++) {
						float weight = softMask.at<float>(i, j);
						weight = min(1.0f, weight * 1.5f); // 가중치 강화

						cv::Vec3b& target = displayFace.at<cv::Vec3b>(wideMouth.y + i, wideMouth.x + j);
						cv::Vec3b blurP = patch.at<cv::Vec3b>(i, j);

						for (int c = 0; c < 3; c++) {
							target[c] = (uchar)(blurP[c] * weight + target[c] * (1.0f - weight));
						}
					}
				}
			}
			displayFace.copyTo(currentFace);
			break;
		}
		case FACE_EMPTY:
		{
			cv::Mat displayFace = currentFace.clone();
			cv::Mat src = originalFace.clone();
			cv::Mat gray, canny;

			// 1. 전처리 및 에지 추출 (목선까지 잡기 위해 50, 50 유지)
			cv::GaussianBlur(src, src, cv::Size(5, 5), 0);
			cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
			cv::Canny(gray, canny, 50, 50);

			// 2. 피부색 마스크 (머리카락 제거용)
			cv::Mat ycrcb, skinMask;
			cv::cvtColor(src, ycrcb, cv::COLOR_BGR2YCrCb);
			cv::inRange(ycrcb, cv::Scalar(0, 133, 77), cv::Scalar(255, 173, 127), skinMask);

			// 3. 피부 영역 내의 에지만 추출
			cv::Mat faceEdges;
			cv::bitwise_and(canny, skinMask, faceEdges);

			// 4. [수정] 마스크 생성 및 목 영역 제외
			std::vector<cv::Point> edgePoints;
			int minY = src.rows, maxY = 0;

			for (int y = 0; y < faceEdges.rows; y++) {
				for (int x = 0; x < faceEdges.cols; x++) {
					if (faceEdges.at<uchar>(y, x) == 255) {
						edgePoints.push_back(cv::Point(x, y));
						if (y < minY) minY = y;
						if (y > maxY) maxY = y;
					}
				}
			}

			cv::Mat internalMask = cv::Mat::zeros(src.size(), CV_8UC1);
			if (!edgePoints.empty()) {
				std::vector<cv::Point> hull;
				cv::convexHull(edgePoints, hull);
				cv::drawContours(internalMask, std::vector<std::vector<cv::Point>>{hull}, -1, cv::Scalar(255), -1);

				// [핵심] 목 보호 로직: 마스크의 하단 20~30% 영역을 점진적으로 지움
				// 보통 얼굴 하단에서 목이 시작되므로 y좌표 기준으로 제한을 둠
				int faceHeight = maxY - minY;
				int neckStartLine = minY + (faceHeight * 0.97); // 얼굴 전체 높이의 75% 지점부터 목으로 간주

				for (int y = neckStartLine; y < src.rows; y++) {
					// 목 부분 마스크를 검은색으로 밀어버림
					internalMask.row(y).setTo(cv::Scalar(0));
				}

				// 안쪽 까만 부분 공략을 위한 침식
				cv::erode(internalMask, internalMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)), cv::Point(-1, -1), 2);
			}

			// 5. 합성 (얼굴 내부만 블러)
			cv::Mat blurred;
			cv::GaussianBlur(displayFace, blurred, cv::Size(171, 171), 0);

			// 3. [핵심] 거리 변환을 이용한 세밀한 가중치 조절
			cv::Mat dist;
			cv::distanceTransform(internalMask, dist, cv::DIST_L2, 3);
			cv::normalize(dist, dist, 0, 1.0, cv::NORM_MINMAX);

			// 5. 최종 합성 (가중치 곡선 조절)
			float power = 0.1f; // [조절 변수] 0.5: 넓게 블러, 1.0: 기본, 2.0: 좁고 중심만 블러

			for (int y = 0; y < displayFace.rows; y++) {
				for (int x = 0; x < displayFace.cols; x++) {
					float d = dist.at<float>(y, x);
					if (d > 0) {
						// d값(중심거리)에 지수를 취해서 블러가 퍼지는 모양을 바꿈
						float weight = std::pow(d, power);

						cv::Vec3b& original = displayFace.at<cv::Vec3b>(y, x);
						cv::Vec3b target = blurred.at<cv::Vec3b>(y, x);

						for (int c = 0; c < 3; c++) {
							original[c] = cv::saturate_cast<uchar>(target[c] * weight + original[c] * (1.0f - weight));
						}
					}
				}
			}
			displayFace.copyTo(currentFace);
			break;
		}
		}

		 effectIndex = (effectIndex + 1) % totalEffects;
	}

	void update() {
		// [수정] 부들부들 떨리거나 크기가 변하는 로직 삭제
		displayRadius = radius; // 항상 고정된 반지름 유지

		if (effectTimer > 0) {
			effectTimer--;
		}
	}

	void draw(cv::Mat& frame) {
		// 1. 현재 표시용 리사이즈
		cv::Mat resizedFace, resizedMask;
		cv::resize(currentFace, resizedFace, cv::Size(displayRadius * 2, displayRadius * 2));
		cv::resize(mask, resizedMask, cv::Size(displayRadius * 2, displayRadius * 2));

		int x = (int)position.x - displayRadius;
		int y = (int)position.y - displayRadius;

		// 2. 화면 영역 이탈 방지
		cv::Rect roi(x, y, resizedFace.cols, resizedFace.rows);
		cv::Rect frameRect(0, 0, frame.cols, frame.rows);
		cv::Rect intersection = roi & frameRect; // 실제 화면과 겹치는 부분만 계산

		if (intersection.width > 0 && intersection.height > 0) {
			// 3. [해결] copyTo에 마스크를 넣으면 마스크가 검은색(0)인 부분은 그리지 않음
			// 검은색 배경은 무시되고 찌그러진 얼굴만 투명하게 합성됨
			cv::Mat subMask = resizedMask(cv::Rect(intersection.x - roi.x, intersection.y - roi.y, intersection.width, intersection.height));
			resizedFace(cv::Rect(intersection.x - roi.x, intersection.y - roi.y, intersection.width, intersection.height))
				.copyTo(frame(intersection), subMask);
		}
	}
};

std::string getEffectName(int index) {
	switch (index) {
	case EMBOSSING:     return "EMBOSSING FILTER";
	case SHARPENING:    return "SHARPENING FILTER";
	case NOISE:         return "GAUSSIAN NOISE ADDED";
	case INVERT:        return "BITWISE INVERT";
	case GRAYSCALE:     return "COLOR TO GRAYSCALE";
	case SEPIA:         return "SEPIA TONE TRANSFORM";
	case COLOR_SWAP:    return "BGR TO RGB SWAP";
	case INVERSE:       return "COLOR INVERSE";
	case FLIP:          return "FLIP & MIRROR";
	case AFFINE_COMBO:  return "AFFINE (MOVE/ROTATE/SHEAR)";
	case PERSPECTIVE:   return "PERSPECTIVE TRANSFORM";
	case RESIZE_ACTION: return "INTERPOLATION RESIZE";
	case MORPH:         return "MORPHOLOGY CLOSE";
	case EYE_EMPTY:     return "EYE DETECT & BACKPROJECT";
	case MOUTH_EMPTY:   return "MOUTH DETECT & MORPHOLOGY";
	case FACE_EMPTY:    return "CANNY EDGE & CONVEX HULL";
	default:            return "NONE";
	}
}
//std::string getEffectName(int index) {
//	switch (index) {
//	case EMBOSSING:     return "EMBOSSING (엠보싱 필터링)";
//	case SHARPENING:    return "SHARPENING (샤프닝/선명화)";
//	case NOISE:         return "GAUSSIAN NOISE (랜덤 잡음 추가)";
//	case INVERT:        return "INVERT (비트 단위 색상 반전)";
//	case GRAYSCALE:     return "GRAYSCALE (명암도 변환)";
//	case SEPIA:         return "SEPIA (세피아 톤 변환)";
//	case COLOR_SWAP:    return "COLOR SWAP (BGR to RGB 채널 변환)";
//	case INVERSE:       return "INVERSE (보색 대비 반전)";
//	case FLIP:          return "FLIP (상하좌우 대칭 변환)";
//	case AFFINE_COMBO:  return "AFFINE (회전+이동+전단 축이동 변환)";
//	case PERSPECTIVE:   return "PERSPECTIVE (원근 투시 변환)";
//	case RESIZE_ACTION: return "RESIZE (보간법 기반 크기 변환)";
//	case MORPH:         return "MORPHOLOGY (모폴로지 닫기 연산)";
//	case EYE_EMPTY:     return "HAAR CASCADE + BACKPROJECT (눈 제거/피부 역투영)";
//	case MOUTH_EMPTY:   return "MOUTH DETECT + MORPHOLOGY (입 제거/모폴로지+이진화)";
//	case FACE_EMPTY:    return "CANNY + CONVEX HULL (얼굴 블러/에지 검출+외곽선)";
//	default:            return "NONE";
//	}
//}

void runProject() CV_NOEXCEPT { // 경고가 뜨기 때문에 CV_NOEXCEPT 쓰는게 좋다
	srand((unsigned int)time(0));
	// Mac 에서는 0번이 안 되면 1번으로 바꿔봐야 할 수도 있음

	cv::VideoCapture cap(0);
	if (!cap.isOpened()) {
		std::cerr << "웹캠이 없습니다.\n";
		return;
	}

	cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

	int width = cvRound(cap.get(cv::CAP_PROP_FRAME_WIDTH));
	int height = cvRound(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

	cv::CascadeClassifier face_cascade;
	std::string xmlFile = "haarcascade_frontalface_default.xml";

	if (!face_cascade.load(xmlFile)) {
		face_cascade.load("C:/OpenCV/sources/data/haarcascades_cuda/" + xmlFile);
	}


	cv::Mat selectScreen = cv::Mat::zeros(cv::Size(640, 480), CV_8UC3);
	cv::Mat myFace;

	// 1. 이미지 로드
	cv::Mat loadedImg = cv::imread("face.jpg");

	if (loadedImg.empty()) {
		std::cout << "face.jpg 파일을 찾을 수 없습니다!" << std::endl;
		// 파일이 없으면 기존처럼 등록 모드로 넘어가거나 종료 처리
	}
	else {
		// 2. 로드된 이미지에서 얼굴 검출
		cv::Mat gray;
		cv::cvtColor(loadedImg, gray, cv::COLOR_BGR2GRAY);

		std::vector<cv::Rect> faces;
		face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(100, 100));

		if (!faces.empty()) {
			// 3. 첫 번째로 검출된 얼굴 영역만 잘라서 myFace에 고정
			myFace = loadedImg(faces[0]).clone();

			// [중요] 공 크기에 맞춰 미리 리사이즈 (떨림/크기변화 방지)
			// 공 반지름이 50이라면 100x100 크기가 적당함
			cv::resize(myFace, myFace, cv::Size(100, 100));

			std::cout << "face.jpg에서 얼굴 추출 성공! 이제 이 얼굴로 고정됨." << std::endl;
			g_mode = CAM_MODE; // 등록 단계를 건너뛰기 위해 모드 강제 설정
		}
		else {
			std::cout << "face.jpg에서 얼굴을 찾지 못했어." << std::endl;
		}
	}

	// 객체 생성 시 이미지 경로와 반지름 전달
	FaceBall redBall(myFace, 40);
	redBall.position = getRandomPosition(width, height, redBall.radius); // 초기 위치 설정

	// 비디오 저장 설정 (XVID 코덱 사용, 파일명 output_YYYYMMDD_HHMMSS.avi)
	std::time_t t = std::time(nullptr);
	std::tm tm = *std::localtime(&t);
	std::stringstream ss;
	ss << "output_" << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".avi";
	std::string videoFileName = ss.str();

	double fps = 20.0; // 녹화용 FPS 설정

	// 비디오 코덱 설정 분기
	cv::VideoWriter writer;
	int fourcc;
	fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D'); // 윈도우

	writer.open(videoFileName, fourcc, fps, cv::Size(width, height));

	if (!writer.isOpened()) {
		std::cerr << "비디오 파일을 열 수 없습니다. 코덱이 설치되었는지 확인하세요.\n";
	}

	cv::Mat frame, prev_gray, standbyImage = cv::imread("start.jpg");
	if (standbyImage.empty()) standbyImage = cv::Mat::zeros(cv::Size(width, height), CV_8UC3);
	else cv::resize(standbyImage, standbyImage, cv::Size(width, height));

	bool isStarted = false, isPaused = false;
	int score = 0;

	cv::Mat pausedFrame; // 일시정지 시점의 화면을 담을 변수

	while (true) {
		cap >> frame;
		if (frame.empty()) break;

		// 반전
		// 영상 이미지가 앞에서 치는것처럼 됨, 그래서 flip 하는게 좋다
		cv::flip(
			frame,
			frame,
			1 // 1 - 좌우반전, 0 - 상하반전, -1 - 좌우 + 상하 반전
		); // 거울 이미지로 반전

		
		int key = cv::waitKey(1) & 0xFF; // 하위 8비트만 남겨서 정확한 키값 추출
		if (key == 27) break;
		isStarted = true;
		
		cv::Mat gray_frame, diff, thresh;
		
		cv::cvtColor(
			frame, // 원본 컬러 이미지 (BGR)
			gray_frame, // 결과물을 저장할 변수 (흑백)
			cv::COLOR_BGR2GRAY // BGR 색상 체계를 GRAY 로 바꿈
		); // 컬러에서 흑백으로 변환

		cv::GaussianBlur( // 손가락 때문에 가우시안 블러까지 씀
			gray_frame, // 입력과 출력을 동시에 수행한다. 흑백 영상을 받아서 다시 그 자리에 뿌연 영상을 저장한다.
			gray_frame, // 입력과 출력을 동시에 수행한다. 흑백 영상을 받아서 다시 그 자리에 뿌연 영상을 저장한다.
			cv::Size(15, 15), // 블러의 강도다. 숫자가 클수록 더 많이 뭉개진다. (반드시 홀수여야 함)
			0 // 표준 편차값이다. 0으로 설정하면 OpenCV가 Size에 맞춰서 알아서 계산해준다.
		);
		

		if (prev_gray.empty()) {
			// 현재 웹캠에서 들어온 첫 번째 흑백 화면(gray_frame)을 prev_gray에 복사해 둔다.
			gray_frame.copyTo(prev_gray); // 이렇게 해야지 끊김없이 부드럽게 넘어감
			continue; // 이번 루프(반복문)의 남은 코드(차이 계산, 공 그리기 등)를 실행하지 말고 맨 처음으로 돌아가라
		}

		// absdiff(): 명암과 객체 분리, 움직임
		// absdiff(): 이전 프레임과 현재 프레임의 차이점(움직임)을 찾음
		cv::absdiff(prev_gray, gray_frame, diff); // 현재 프레임, 그레이 프레임 비교 후 diff 에 넣기 (차이객체를 diff에다 넣는다)
		
		// 그 희미한 회색들이 선명한 흰색 덩어리가 됨 - 이진화
		cv::threshold(
			diff, 
			thresh, 
			25.0, // 픽셀의 차이가 25보다 작으면 노이즈로 간주하고 0(검은색)으로 만든다. 25보다 크면 움직임이 확실하다고 판단
			255.0, // 임계값을 넘은 픽셀들을 어떤 색으로 바꿀지 정한다. 255는 완전한 흰색임
			cv::THRESH_BINARY // 이진화(Binary) 모드다. 즉, 중간 단계 없이 검은색(0) 아니면 흰색(255) 둘 중 하나로만 표현하겠다는 설정이다.
		); // 25 ~ 255

		// 임계값에 의해 redBall의 activie 추가되면
		// 공이 활성화된 상태에서만 터치 감지 로직 실행
		if (redBall.active) { // 공이 activity 한다는 뜻
			//  얼굴 영역(ROI) 설정
			cv::Rect ballRect(
				redBall.position.x - redBall.radius,
				redBall.position.y - redBall.radius,
				redBall.radius * 2,
				redBall.radius * 2
			);
			cv::Rect validBallRect = ballRect & cv::Rect(0, 0, width, height);

			// 경계 검사
			// 이 연산을 수행하면 ballRect는 화면 범위(0, 0, width, height) 안의 영역만 남기고 나머지는 버린다.
			// 경계 검사: ballRect가 화면 밖으로 나가지 않도록 조정 (교집합)
			if (validBallRect.area() > 0) {
				cv::Mat roi = thresh(validBallRect);

				if (countNonZero(roi) > (validBallRect.area() * 0.1)) {
					score++;
					redBall.applyRandomEffect(width, height);

					// [추가] 공을 화면 밖으로 치워버리고 '비활성화' 시킴
					redBall.active = false;
					redBall.position = cv::Point(-500, -500);
					spawnTimer = 0; // 타이머 리셋
				}
			}
		}

		if (!redBall.active) {
			spawnTimer++; // 공이 사라진 동안 숫자 증가

			if (spawnTimer >= spawnDelay) {
				// [부활] 시간이 다 되면 공을 새 위치에 소환
				redBall.position = getRandomPosition(width, height, redBall.radius);
				redBall.active = true;
				spawnTimer = 0;

				// 자막 초기화 (공이 새로 나오면 자막 지우고 싶을 때)
				// g_currentEffectIdx = -1; 
			}
		}

		// 공의 애니메이션 (크기, 색상) 업데이트
		redBall.update();
		redBall.draw(frame);

		cv::putText(
			frame, // 글자를 써넣을 대상 이미지 변수
			"Score : " + std::to_string(score), // 화면에 표시할 문자열
			cv::Point(20, 50), // 글자가 시작될 좌표
			cv::FONT_HERSHEY_PLAIN, // 폰트 종류
			2, // 폰트 크기
			cv::Scalar(255, 255, 255), // 글자 색상
			2 // 글자 두께
		);

		// [수정] 자막 출력 부분
		// 공이 비활성화 상태(!active)이거나 효과 타이머가 작동 중일 때 자막 표시
		if ((!redBall.active || redBall.effectTimer > 0) && g_currentEffectIdx != -1) {
			std::string effectText = "EFFECT: " + getEffectName(g_currentEffectIdx);

			// 자막 가독성을 위해 검은색 외곽선(그림자)
			cv::putText(frame, effectText, cv::Point(32, height - 48),
				cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 4);
			// 그 위에 흰색 글씨
			cv::putText(frame, effectText, cv::Point(30, height - 50),
				cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
		}

		// 프레임 저장
		if (writer.isOpened()) {
			writer.write(frame);
		}

		cv::imshow("GAME", frame);

		gray_frame.copyTo(prev_gray); // 화면에 업데이트 (부드럽게 하기 위해)
	}
	cap.release();
	writer.release();
	cv::destroyAllWindows(); // 모든 openCV 창 닫기
}

int main() {
	initGameSounds();
	runProject();
	return 0;
}
