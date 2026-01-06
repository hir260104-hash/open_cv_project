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
}

void playHitSound() {
	mciSendStringA("play hit_sound from 0", NULL, 0, NULL);
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
	cv::Point position;    // 공의 중심 좌표
	int radius;            // 기본 반지름
	int displayRadius;     // 애니메이션용(커지는 효과 등) 반지름
	cv::Mat originalFace;  // 변형 전 원본 얼굴 이미지
	cv::Mat currentFace;   // 현재 효과가 적용된 얼굴 이미지
	cv::Mat originalMask;  // 원본 마스크 (원형 또는 사각형)
	cv::Mat mask;          // 변형(회전/투시)이 적용된 마스크
	int hitCount = 0;      // 타격 횟수 (누적될수록 공이 빨개짐)
	int effectTimer = 0;   // 효과 자막 유지 시간
	bool active = true;    // 현재 화면에 표시되고 있는지 여부

	// 생성자: 이미지 경로를 받아 공 생성
	FaceBall(std::string path, int r) { // 생성자, Ball의 객체
		radius = r;
		displayRadius = r;
		originalFace = cv::imread(path);

		if (originalFace.empty()) { // 이미지 로드 실패 시 빨간 원 생성
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

	// 생성자: 캡처된 Mat 객체를 직접 받아 공 생성 (사각형 마스크 사용)
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

	// [핵심] 공을 때렸을 때 랜덤한 시각 효과 적용
	void applyRandomEffect(int width, int height) {
		hitCount++; // 때릴 때마다 누적

		playHitSound();

		g_currentEffectIdx = effectIndex;

		int totalEffects = COUNT; // 총 효과 개수
		
		// [핵심] 누적 방지 로직
		// 이전 타격 때 적용했던 효과(예: 흑백) 위에 새로운 효과(예: 회전)를 덧칠하면 이미지가 망가짐
		// 그래서 매번 타격 때마다 "깨끗한 원본"을 다시 복사해와서 새로 시작함
		currentFace = originalFace.clone();
		mask = originalMask.clone();

		// 타격 누적에 따라 점점 빨간색 추가
		currentFace += cv::Scalar(0, 0, min(hitCount * 10, 255));

		effectTimer = 15; // 효과 유지 프레임 수
		displayRadius = radius * 1.5; // 터치 시 기본적으로 커짐 (크기변환)

		// 원본 얼굴 이미지의 가로(wf), 세로(hf) 크기를 실수형으로 저장 (수학 계산용)
		float wf = (float)originalFace.cols;
		float hf = (float)originalFace.rows;

		// 변환 시 잘림 방지를 위한 공통 설정
		// BORDER_REFLECT_101: 경계선을 기준으로 거울처럼 반사하여 빈 곳을 채움
		int borderMode = cv::BORDER_REFLECT_101;

		switch (effectIndex) {
		case EMBOSSING: // 엠보싱 효과: 이미지를 조각한 것처럼 입체감을 줌
		{
			// 커널(Kernel): 이미지 위에 올려놓고 계산하는 3x3 크기의 작은 수학 필터
			cv::Mat kernel = (cv::Mat_<float>(3, 3) << -1, -1, 0, -1, 0, 1, 0, 1, 1);

			// filter2D: 이 수학 필터를 이미지 전체에 훑으면서 적용하는 함수
			cv::filter2D(currentFace, currentFace, -1, kernel, cv::Point(-1, -1), 128);
			break;
		}
		case SHARPENING:// 샤프닝 효과: 흐릿한 이미지를 선명하게 만듦
		{
			// 중심 값을 높이고 주변을 깎아내는 필터를 사용하여 경계선을 뚜렷하게 함
			cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
			cv::filter2D(currentFace, currentFace, -1, kernel);
			break;
		}
		case NOISE: // 노이즈 효과: 화면에 지지직거리는 잡음을 섞음
		{
			cv::Mat noise(currentFace.size(), currentFace.type());
			// randn: 평균 0, 표준편차 100의 가우시안(정규분포) 랜덤 값을 생성함
			cv::randn(noise, 0, 100);
			currentFace += noise; // randn: 평균 0, 표준편차 100의 가우시안(정규분포) 랜덤 값을 생성함
			break;
		}
		case INVERT: // 색상 반전: 검은색은 흰색으로, 빨간색은 청록색으로 뒤집음
			cv::bitwise_not(currentFace, currentFace); // 비트 단위 NOT 연산 (색상 반전)
			break;
		case GRAYSCALE: // 흑백 변환: 모든 색상을 명암으로만 표현
		{
			cv::Mat gray;
			cv::cvtColor(currentFace, gray, cv::COLOR_BGR2GRAY); // 컬러를 흑백으로
			cv::cvtColor(gray, currentFace, cv::COLOR_GRAY2BGR); // 다시 3채널로 바꿔야 다른 연산과 호환됨
			break;
		}
		case SEPIA: // 세피아 효과: 옛날 빛바랜 사진처럼 황갈색 톤으로 변환
		{
			// 세피아 톤을 만드는 특정 행렬 값을 정의함
			cv::Mat m = (cv::Mat_<float>(3, 3) <<
				0.131, 0.534, 0.272,
				0.168, 0.686, 0.349,
				0.189, 0.769, 0.393);
			cv::transform(currentFace, currentFace, m); // 행렬 연산을 이미지에 적용
			break;
		}
		case COLOR_SWAP: // 색상 채널 교체: 파란색과 빨간색을 서로 바꿈
			cv::cvtColor(currentFace, currentFace, cv::COLOR_BGR2RGB);
			break;
		case INVERSE:
			// 원본 얼굴의 색상을 완전히 뒤집기
			currentFace = ~originalFace;
			break;
		case FLIP: // 대칭 및 반전: 상하좌우를 거울처럼 뒤집음
		{
			cv::flip(currentFace, currentFace, -1); // -1은 상하좌우 모두 반전
			cv::flip(mask, mask, -1); // 마스크(공의 영역)도 같이 뒤집어야 함
			break;
		}
		case AFFINE_COMBO: // 어파인 변환: 회전 + 크기 조절 + 밀림 효과를 동시에 적용
		{
			cv::Point2f center(wf / 2.0f, hf / 2.0f); // 이미지의 정중앙 지점
			double angle = (rand() % 61) - 30; // -30도 ~ +30도 사이 랜덤 회전
			double scale = 0.5 + (rand() % 11) / 100.0; // 이미지가 잘리지 않게 크기를 줄임

			// 회전과 크기를 계산한 변환 행렬 M을 만듦
			cv::Mat M = cv::getRotationMatrix2D(center, angle, scale);

			// 축 이동 범위를 좁혀서 경계선 탈출 방지
			M.at<double>(0, 2) += (rand() % 7 - 3);
			M.at<double>(1, 2) += (rand() % 7 - 3);

			// 찌그러짐 (Shear) 적용
			M.at<double>(0, 1) += ((rand() % 21) - 10) / 100.0;
			M.at<double>(1, 0) += ((rand() % 21) - 10) / 100.0;

			// warpAffine: 계산된 행렬 M을 바탕으로 이미지를 실제로 비틂
			cv::warpAffine(currentFace, currentFace, M, currentFace.size(),
				cv::INTER_LINEAR);
			cv::warpAffine(mask, mask, M, mask.size(),
				cv::INTER_LINEAR); // 마스크도 함께 회전해야 잘리지 않음
			break;
		}
		case PERSPECTIVE: // 투시 변환: 이미지를 비스듬하게 눕히거나 입체적으로 왜곡함
		{
			// 원본의 네 모서리 좌표
			cv::Point2f srcPts[4] = { {0.f, 0.f}, {wf - 1.f, 0.f}, {wf - 1.f, hf - 1.f}, {0.f, hf - 1.f} };
			cv::Point2f dstPts[4];

			// 결과물이 나타날 좌표 (일부러 안쪽으로 모아서 찌그러뜨림)
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

			// warpPerspective: 3D 공간에서 보는 것처럼 이미지를 투시 왜곡함
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
		case MORPH: // 모폴로지 기법: 사물의 형태를 뭉개거나 확장함
		{
			// 둥근 모양의 필터 커널 생성
			cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));

			// MORPH_CLOSE: 구멍난 부분을 메우고 외곽선을 뭉뚱그려 눈코입이 뭉쳐지게 함
			morphologyEx(originalFace, currentFace, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 5);
			break;
		}
		case EYE_EMPTY: // 눈 제거 효과 (딥러닝 기반 아님, 전통적 필터링)
		{
			// 1. 이미지 복사 및 전처리
			cv::Mat displayFace = currentFace.clone(); // 실제 화면에 보여줄(변형할) 이미지
			cv::Mat maskSource = originalFace.clone(); // 분석용 원본 이미지
			cv::Mat ycrcb, gray;

			// 분석을 위해 색상 체계 변경
			cv::cvtColor(maskSource, ycrcb, cv::COLOR_BGR2YCrCb); // 피부색 판별에 유리한 YCrCb 컬러
			cv::cvtColor(maskSource, gray, cv::COLOR_BGR2GRAY); // 눈 위치 검출을 위한 흑백(Gray)

			// 2. 눈 검출 (Cascade Classifier)
			cv::CascadeClassifier eyeCascade("haarcascade_eye.xml"); // 미리 학습된 눈 검출기 로드
			std::vector<cv::Rect> eyes;

			// 이미지에서 눈을 찾아 좌표(Rect) 리스트를 생성함
			eyeCascade.detectMultiScale(gray, eyes, 1.1, 3, 0, cv::Size(10, 10));

			// 검출된 눈 영역들을 저장할 빈 검은색 마스크 판 (전체 크기)
			cv::Mat fullMask = cv::Mat::zeros(maskSource.size(), CV_8UC1);

			// 찾은 눈의 개수만큼 반복 실행
			for (cv::Rect eye : eyes) {
				// 3. 눈이 있는 자리를 피부색으로 칠하고 블러(Blur) 처리해서 자연스럽게 지움
				// (핵심 로직: 주변 피부색 샘플링 -> 역투영으로 눈 영역 분리 -> 덮어쓰기)
	
				// --- [A] 피부색 샘플링 단계 ---
				// 눈 주변(눈썹 위 등) 아주 좁은 영역(3x3)을 피부색 샘플로 지정함
				cv::Rect skinROI(eye.x, eye.y, 3, 3);
				if (skinROI.y < 0 || skinROI.y >= ycrcb.rows) continue;

				cv::Mat sample = ycrcb(skinROI); // 샘플 영역 추출
				cv::Scalar avgSkin = cv::mean(maskSource(skinROI));
				cv::Vec3b skinColor(avgSkin[0], avgSkin[1], avgSkin[2]);

				// 히스토그램(색상 분포도) 계산: 샘플링한 피부색이 어떤 수치인지 분석함
				cv::Mat hist;
				int channels[] = { 1, 2 }; // Y(밝기)를 제외한 Cr, Cb(색상 정보)만 사용
				int histSize[] = { 32, 32 };
				float range[] = { 0, 256 };
				const float* ranges[] = { range, range };
				cv::calcHist(&sample, 1, channels, cv::Mat(), hist, 2, histSize, ranges);
				cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX); // 수치 표준화

				// --- [B] 눈 영역 분리 (역투영) ---
				// calcBackProject: "아까 분석한 피부색과 다른 부분이 어디인가?"를 찾음
				cv::Mat eyeBP;
				cv::Mat eyeROI = ycrcb(eye);
				cv::calcBackProject(&eyeROI, 1, channels, hist, eyeBP, ranges);

				// 피부색이 아닌 부분(눈동자, 속눈썹 등)을 흰색으로 추출 (이진화)
				cv::Mat eyeMask;
				cv::threshold(eyeBP, eyeMask, 60, 255, cv::THRESH_BINARY_INV);

				// --- [C] 형태학적 보정 (모폴로지) ---
				// 둥근 모양의 연산 도구 생성
				cv::Mat morphKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));

				// 소소한 구멍들을 메우고 (CLOSE)
				cv::morphologyEx(eyeMask, eyeMask, cv::MORPH_CLOSE, morphKernel, cv::Point(-1, -1), 3);
				// 흰색 영역을 사방으로 확 키워서 눈썹과 주변 그림자를 포함시킴 (DILATE)
				cv::morphologyEx(eyeMask, eyeMask, cv::MORPH_DILATE, morphKernel, cv::Point(-1, -1), 2);

				// 해당 눈 영역의 결과만 fullMask에 업데이트
				eyeMask.copyTo(fullMask(eye));

				// --- [D] 부드러운 합성(Blending) 준비 ---
				int p = 15; // 패딩 복구 (경계선 제거 필수)
				cv::Rect wideEye; // 눈보다 조금 더 넓은 영역 설정
				wideEye.x = max(0, eye.x - p);
				wideEye.y = max(0, eye.y - p);
				wideEye.width = min(displayFace.cols - wideEye.x, eye.width + 2 * p);
				wideEye.height = min(displayFace.rows - wideEye.y, eye.height + 2 * p);

				// 현재 타격으로 인해 빨개진 피부색 등을 반영하기 위해 실시간 평균 색상 추출
				cv::Scalar currentAvg = cv::mean(displayFace(wideEye));
				cv::Vec3b dynamicSkinColor((uchar)currentAvg[0], (uchar)currentAvg[1], (uchar)currentAvg[2]);

				cv::Mat patch = displayFace(wideEye).clone();
				cv::Mat localMask = fullMask(wideEye);

				// 마스크 경계선을 부드럽게 깎기 위해 가우시안 블러 적용
				cv::Mat softMask;
				cv::GaussianBlur(localMask, softMask, cv::Size(21, 21), 0);
				softMask.convertTo(softMask, CV_32F, 1.0 / 255.0); // 0~1 사이 값으로 변환

				// --- [E] 색칠 및 블러 처리 ---
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
				// 칠해진 조각(patch)을 아주 강하게 뭉개서 피부 결처럼 보이게 함
				cv::GaussianBlur(patch, patch, cv::Size(51, 51), 0);

				// --- [F] 최종 합성 (알파 블렌딩) ---
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
			// 1. 준비 단계
			cv::Mat displayFace = currentFace.clone(); // 실제 변형을 가할 이미지 (빨간 피드백 포함)
			cv::Mat maskSource = originalFace.clone(); // 입 위치를 찾기 위한 깨끗한 원본 이미지

			cv::Mat debugImg = maskSource.clone();     // 개발자가 확인하기 위해 사각형 등을 그릴 이미지
			cv::Mat ycrcb, gray;
			cv::cvtColor(maskSource, ycrcb, cv::COLOR_BGR2YCrCb); // 피부색 판별용 (색상 중심)
			cv::cvtColor(maskSource, gray, cv::COLOR_BGR2GRAY);   // 입 검출용 (명암 중심)

			// 2. 입 검출기 설정
			// OpenCV에서 제공하는 입 전용 학습 모델을 불러옴
			cv::CascadeClassifier mouthCascade("haarcascade_mcs_mouth.xml");
			if (mouthCascade.empty()) {
				std::cout << "모델 로드 실패! 경로 확인." << std::endl;
			}

			std::vector<cv::Rect> mouths;
			// detectMultiScale: 이미지 안에서 입의 사각형 좌표들을 찾아냄
			// 1.3: 검색 스케일, 5: 인접 후보 개수 (값이 클수록 엄격하게 검출)
			mouthCascade.detectMultiScale(gray, mouths, 1.3, 5, 0, cv::Size(20, 20));

			// 검출된 입 영역들을 기록할 전체 크기의 빈 도화지(마스크)
			cv::Mat fullMask = cv::Mat::zeros(maskSource.size(), CV_8UC1);

			for (cv::Rect mouth : mouths) {
				// [입 전용 샘플링] 입술색을 피하기 위해 입 사각형 바로 위(인중 근처) 피부를 조준
				// mouth.y - 10 지점에서 가로로 길게 샘플을 따서 현재 피부색을 파악함
				cv::Rect skinROI(mouth.x + mouth.width / 4, max(0, mouth.y - 10), mouth.width / 2, 5);
				if (skinROI.y < 0 || skinROI.y >= ycrcb.rows) continue;

				// --- [색상 분석] ---
				cv::Mat sample = ycrcb(skinROI); // 샘플링한 피부 영역
				cv::Mat hist;
				int channels[] = { 1, 2 }; // 밝기를 뺀 나머지 색상 정보만 사용
				int histSize[] = { 32, 32 };
				float range[] = { 0, 256 };
				const float* ranges[] = { range, range };

				// 추출한 피부색이 어떤 분포를 가지는지 '성적표(히스토그램)'를 만듦
				cv::calcHist(&sample, 1, channels, cv::Mat(), hist, 2, histSize, ranges);
				cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX);

				// --- [입 영역 분리] ---
				// 입이 있는 곳(mouthROI)에서 아까 만든 피부색 성적표와 대조함
				cv::Mat mouthROI = ycrcb(mouth);
				cv::Mat mouthBP;
				cv::calcBackProject(&mouthROI, 1, channels, hist, mouthBP, ranges);

				// 피부색과 너무 다른 부분(입술 안쪽, 그림자 등)을 흰색으로 추출
				cv::Mat mouthMask;
				cv::threshold(mouthBP, mouthMask, 60, 255, cv::THRESH_BINARY_INV);

				// --- [강력 보정] ---
				// 입은 눈보다 면적이 넓고 굴곡이 심하므로 더 큰 커널(9x9)을 사용
				cv::Mat morphKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
				// CLOSE: 입술 사이의 틈이나 치아 부분을 메움
				cv::morphologyEx(mouthMask, mouthMask, cv::MORPH_CLOSE, morphKernel, cv::Point(-1, -1), 2);
				// DILATE: 지울 영역을 입술 주변까지 넉넉하게 확장
				cv::dilate(mouthMask, mouthMask, morphKernel, cv::Point(-1, -1), 2);

				// 결과물을 전체 마스크 판의 입 위치에 복사
				mouthMask.copyTo(fullMask(mouth));

				// --- [자연스러운 합성] ---
				int p = 25; // 입 주변 살점을 넉넉히 가져오기 위한 패딩
				cv::Rect wideMouth;
				wideMouth.x = max(0, mouth.x - p);
				wideMouth.y = max(0, mouth.y - p);
				wideMouth.width = min(displayFace.cols - wideMouth.x, mouth.width + 2 * p);
				wideMouth.height = min(displayFace.rows - wideMouth.y, mouth.height + 2 * p);

				// 현재 타격으로 인해 빨개진 상태의 평균 색상을 가져옴
				cv::Scalar currentAvg = cv::mean(displayFace(wideMouth));
				cv::Vec3b dynamicSkinColor((uchar)currentAvg[0], (uchar)currentAvg[1], (uchar)currentAvg[2]);

				cv::Mat patch = displayFace(wideMouth).clone(); // 덮어씌울 피부 조각
				cv::Mat localMask = fullMask(wideMouth);

				// 마스크 경계를 아주 부드럽게 만들기 위한 블러 (Size 51)
				cv::Mat softMask;
				cv::GaussianBlur(localMask, softMask, cv::Size(51, 51), 0);
				softMask.convertTo(softMask, CV_32F, 1.0 / 255.0);

				// 1. 패치에 피부색 입히기
				for (int i = 0; i < patch.rows; i++) {
					for (int j = 0; j < patch.cols; j++) {
						if (localMask.at<uchar>(i, j) == 255) {
							cv::Vec3b& px = patch.at<cv::Vec3b>(i, j);
							float blendRatio = 0.95f; // 입술색은 진하므로 95% 강도로 덮어버림
							for (int c = 0; c < 3; c++)
								px[c] = px[c] * (1.0f - blendRatio) + dynamicSkinColor[c] * blendRatio;
						}
					}
				}

				// 2. 패치 뭉개기: 입술의 질감을 완전히 없애기 위해 매우 강한 블러(71x71) 적용
				cv::GaussianBlur(patch, patch, cv::Size(71, 71), 0);

				// 3. 최종 알파 블렌딩: 조각을 얼굴 본체에 부드럽게 이식
				for (int i = 0; i < patch.rows; i++) {
					for (int j = 0; j < patch.cols; j++) {
						float weight = softMask.at<float>(i, j);
						weight = min(1.0f, weight * 1.5f); // 중심부는 더 확실하게 지워지도록 가중치 강화

						cv::Vec3b& target = displayFace.at<cv::Vec3b>(wideMouth.y + i, wideMouth.x + j);
						cv::Vec3b blurP = patch.at<cv::Vec3b>(i, j);

						for (int c = 0; c < 3; c++) {
							// 가중치에 따라 원래 얼굴색과 준비한 피부 패치를 섞음
							target[c] = (uchar)(blurP[c] * weight + target[c] * (1.0f - weight));
						}
					}
				}
			}
			// 변형이 완료된 이미지를 실제 얼굴로 업데이트
			displayFace.copyTo(currentFace);
			break;
		}
		case FACE_EMPTY:
		{
			cv::Mat displayFace = currentFace.clone();
			cv::Mat src = originalFace.clone();
			cv::Mat gray, canny;

			// 1. 에지 추출 확인
			cv::GaussianBlur(src, src, cv::Size(5, 5), 0);
			cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
			cv::Canny(gray, canny, 50, 50);

			// 2. 피부색 영역 확인
			cv::Mat ycrcb, skinMask;
			cv::cvtColor(src, ycrcb, cv::COLOR_BGR2YCrCb);
			cv::inRange(ycrcb, cv::Scalar(0, 133, 77), cv::Scalar(255, 173, 127), skinMask);

			// 3. 피부 내 에지 확인
			cv::Mat faceEdges;
			cv::bitwise_and(canny, skinMask, faceEdges);

			// 4. 좌표 수집 및 로그 출력
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
			std::cout << "[LOG] 수집된 에지 점 개수: " << edgePoints.size() << std::endl;
			std::cout << "[LOG] 얼굴 Y축 범위: " << minY << " ~ " << maxY << " (높이: " << (maxY - minY) << ")" << std::endl;

			cv::Mat internalMask = cv::Mat::zeros(src.size(), CV_8UC1);
			if (!edgePoints.empty()) {
				std::vector<cv::Point> hull;

				// 가장 외곽에 있는 점들만 골라내서 순서대로 연결
				// convexHull 은 단순히 외곽 점들의 좌표 목록
				cv::convexHull(edgePoints, hull);

				// hull은 점들의 벡터(std::vector<cv::Point>)이므로, 
				// 이를 다시 중괄호 { }로 감싸서 '벡터의 벡터' 형태로 전달해야 함
				cv::drawContours(
					internalMask, 
					std::vector<std::vector<cv::Point>>{hull}, 
					-1, 
					cv::Scalar(255), 
					-1 // 안쪽 영역을 페인트로 가득 채워라
				);

				// 목 보호 로직 실행
				int faceHeight = maxY - minY;
				int neckStartLine = minY + (faceHeight * 0.97);
				std::cout << "[LOG] 목 시작 예상 라인(Y좌표): " << neckStartLine << std::endl;

				for (int y = neckStartLine; y < src.rows; y++) {
					internalMask.row(y).setTo(cv::Scalar(0));
				}

				cv::erode(internalMask, internalMask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)), cv::Point(-1, -1), 2);
			}

			// 5. 블러 이미지 확인
			cv::Mat blurred;
			cv::GaussianBlur(displayFace, blurred, cv::Size(171, 171), 0);

			// 6. 거리 변환 맵 확인 (가장 중요)
			cv::Mat dist;

			// cv::distanceTransform은 흰색 영역 안의 모든 점에 대해 **"가장 가까운 검은색(경계선)까지의 거리"**를 수치로 바꾼 결과
			// 3 대신 5를 넣으면 더 정밀한 거리 계산을 해서 각진 모양이 사라짐
			// "가운데가 어디지?"라고 먼저 찾는 게 아니야. 반대로 **"검은색(경계선)에서 얼마나 멀어지는가"**를 계산하는 방식
			// 가운데를 미리 아는 게 아니라, 사방의 벽(테두리)으로부터 도망치다 보니 가장 멀리 도망친 곳이 자연스럽게 하이라이트가 되는 원리
			cv::distanceTransform(
				internalMask, 
				dist, 
				cv::DIST_L2, 
				5 // 마스크 사이즈
			);
			cv::normalize(dist, dist, 0, 1.0, cv::NORM_MINMAX);

			// 7. 합성 및 로그
			float power = 0.1f;
			int appliedPixels = 0;

			for (int y = 0; y < displayFace.rows; y++) {
				for (int x = 0; x < displayFace.cols; x++) {
					// 중심부: 테두리에서 머니까 d가 크고, 결과적으로 weight도 커짐.그래서 눈코입이 있는 중심은 확실하게 블러(target)로 덮여서 사라짐.
					// 외곽부 : 테두리에 가까우니 d가 작고 weight도 작음.덕분에 턱선이나 머리카락 경계는 원래 내 얼굴(original)이 그대로 유지되면서 자연스럽게 이어짐.
					float d = dist.at<float>(y, x);
					if (d > 0) {
						appliedPixels++;

						// distanceTransform 이 준 가중치를 그대로 쓰면 지워지는 느낌이 너무 심심하거나 선형적일 수 있다. 그래서 pow 함수를 써서 가중치의 성질을 바꿔버림
						// 그냥 d를 쓸 때: 중심에서 테두리까지 아주 정직하게(직선형으로) 서서히 흐려짐
						// pow(d, 0.1)을 쓸 때: 아주 작은 거리값만 있어도 결과값이 확 커짐
						float weight = std::pow(d, power);
						cv::Vec3b& original = displayFace.at<cv::Vec3b>(y, x);
						cv::Vec3b target = blurred.at<cv::Vec3b>(y, x);

						// 픽셀 단위의 "색상 혼합" (알파 블렌딩)
						// "원본 색상"과 "지워진 색상(블러)"을 몇 대 몇으로 섞을지 결정
						// target(블러) * weight + original(원본) * (1 - weight)
						// weight가 0.9라면 블러를 90%, 원본을 10% 섞으라는 뜻
						for (int c = 0; c < 3; c++) {
							original[c] = cv::saturate_cast<uchar>(target[c] * weight + original[c] * (1.0f - weight));
						}
					}
				}
			}
			std::cout << "[LOG] 실제 합성된 픽셀 수: " << appliedPixels << std::endl;

			displayFace.copyTo(currentFace);
			cv::waitKey(1); // 실시간 확인을 위해 짧은 대기
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
