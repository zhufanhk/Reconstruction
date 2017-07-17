#include "openMVG/cameras/cameras.hpp"
#include "openMVG/image/image.hpp"
#include "openMVG/features/features.hpp"
#include "openMVG/sfm/sfm.hpp"

#include "openMVG/matching/matcher_brute_force.hpp"
#include "openMVG/matching/indMatchDecoratorXY.hpp"
#include "openMVG/multiview/triangulation.hpp"

#include "openMVG/matching/regions_matcher.hpp"
#include "nonFree/sift/SIFT_describer.hpp"

#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"
#include "third_party/vectorGraphics/svgDrawer.hpp"

#include <string>
#include <iostream>
#include <math.h>

using namespace openMVG;
using namespace openMVG::image;
using namespace openMVG::matching;
using namespace openMVG::cameras;
using namespace openMVG::geometry;
using namespace openMVG::sfm;
using namespace svg;
using namespace std;


bool readIntrinsic(const std::string & fileName, Mat3 & K)
{
	// Load the K matrix( F 0 ppx,0 F pp,0 0 1)
	ifstream in;
	in.open(fileName.c_str(), ifstream::in);
	if (in.is_open()) {
		for (int j = 0; j < 3; ++j)
			for (int i = 0; i < 3; ++i)
				in >> K(j, i);
	}
	else {
		std::cerr << std::endl
			<< "Invalid input K.txt file" << std::endl;
		return false;
	}
	return true;
}

bool readIntrinsic(const std::string & fileName, Mat3 & K);

int main() {

	const std::string sInputDir = "E:/ImageDataset/images/";
	Image<RGBColor> image;
	const string jpg_filenameL = sInputDir + "100_7101.jpg";
	const string jpg_filenameR = sInputDir + "100_7102.jpg";

	Image<unsigned char> imageL, imageR;
	ReadImage(jpg_filenameL.c_str(), &imageL);
	ReadImage(jpg_filenameR.c_str(), &imageR);

	//describe the features
	using namespace openMVG::features;
	std::unique_ptr<Image_describer> image_describer(new SIFT_Image_describer);
	std::map<IndexT, std::unique_ptr<features::Regions> > regions_perImage;
	image_describer->Describe(imageL, regions_perImage[0]);
	image_describer->Describe(imageR, regions_perImage[1]);

	const SIFT_Regions* regionsL = dynamic_cast<SIFT_Regions*>(regions_perImage.at(0).get());
	const SIFT_Regions* regionsR = dynamic_cast<SIFT_Regions*>(regions_perImage.at(1).get());

	const PointFeatures
		featsL = regions_perImage.at(0)->GetRegionsPositions(),
		featsR = regions_perImage.at(1)->GetRegionsPositions();

	// concat the images
	Image<unsigned char> concat;
	ConcatH(imageL, imageR, concat);
	//string out_filename1 = "test.jpg";
	//WriteImage(out_filename1.c_str(), concat);



	//draw features :
	for (size_t i = 0; i < featsL.size(); ++i) {
		const SIOPointFeature point = regionsL->Features()[i];
		DrawCircle(point.x(), point.y(), point.scale(), 255, &concat);
	}
	for (size_t i = 0; i < featsR.size(); ++i) {
		const SIOPointFeature point = regionsR->Features()[i];
		DrawCircle(point.x() + imageL.Width(), point.y(), point.scale(), 255, &concat);
	}
	string out_filename2 = "features.jpg";
	WriteImage(out_filename2.c_str(), concat);


	//find the matches
	std::vector<IndMatch> vec_PutativeMatches;
	// Find corresponding points
	matching::DistanceRatioMatch(
		0.8, matching::BRUTE_FORCE_L2,
		*regions_perImage.at(0).get(),
		*regions_perImage.at(1).get(),
		vec_PutativeMatches);

	IndMatchDecorator<float> matchDeduplicator(
		vec_PutativeMatches, featsL, featsR);
	matchDeduplicator.getDeduplicated(vec_PutativeMatches);

	std::cout
		<< regions_perImage.at(0)->RegionCount() << " #Features on image A" << std::endl
		<< regions_perImage.at(1)->RegionCount() << " #Features on image B" << std::endl
		<< vec_PutativeMatches.size() << " #matches with Distance Ratio filter" << std::endl;

	// Draw correspondences after Nearest Neighbor ratio filter
	svgDrawer svgStream(imageL.Width() + imageR.Width(), max(imageL.Height(), imageR.Height()));
	svgStream.drawImage(jpg_filenameL, imageL.Width(), imageL.Height());
	svgStream.drawImage(jpg_filenameR, imageR.Width(), imageR.Height(), imageL.Width());
	for (size_t i = 0; i < vec_PutativeMatches.size(); ++i) {
		//Get back linked feature, draw a circle and link them by a line
		const SIOPointFeature L = regionsL->Features()[vec_PutativeMatches[i].i_];
		const SIOPointFeature R = regionsR->Features()[vec_PutativeMatches[i].j_];
		svgStream.drawLine(L.x(), L.y(), R.x() + imageL.Width(), R.y(), svgStyle().stroke("green", 2.0));
		svgStream.drawCircle(L.x(), L.y(), L.scale(), svgStyle().stroke("yellow", 2.0));
		svgStream.drawCircle(R.x() + imageL.Width(), R.y(), R.scale(), svgStyle().stroke("yellow", 2.0));
	}
	const std::string out_filename3 = "Matches.svg";
	std::ofstream svgFile(out_filename3.c_str());
	svgFile << svgStream.closeSvgFile().str();
	svgFile.close();

	{
	// Essential geometry filtering of putative matches
	Mat3 K;
	//read K from file
	if (!readIntrinsic(stlplus::create_filespec(sInputDir, "K", "txt"), K))
	{
		std::cout << "Cannot read intrinsic parameters." << std::endl;
		return EXIT_FAILURE;
	}

	//else find corresponding points and compute parameters
	Mat xL(2, vec_PutativeMatches.size());
	Mat xR(2, vec_PutativeMatches.size());
	for (size_t k = 0; k < vec_PutativeMatches.size(); ++k) {
		const PointFeature & imaL = featsL[vec_PutativeMatches[k].i_];
		const PointFeature & imaR = featsR[vec_PutativeMatches[k].j_];
		xL.col(k) = imaL.coords().cast<double>();
		xR.col(k) = imaR.coords().cast<double>();
	}

	//relative pose 
	std::pair<size_t, size_t> size_imaL(imageL.Width(), imageL.Height());
	std::pair<size_t, size_t> size_imaR(imageR.Width(), imageR.Height());
	sfm::RelativePose_Info relativePose_info;
	if (!sfm::robustRelativePose(K, K, xL, xR, relativePose_info, size_imaL, size_imaR, 256))
	{
		std::cout << " Robust relative pose estimation failure."
			<< std::endl;
		return EXIT_FAILURE;
	}

	std::cout << "\nresult!!!\n"
		<< "\tinliers: " << relativePose_info.vec_inliers.size() << "\n"
		<< "\tmatches: " << vec_PutativeMatches.size() << "\n"
		<< "\tratio:" << (double(relativePose_info.vec_inliers.size()) / double(vec_PutativeMatches.size()))
		<< std::endl;

	std::cout << std::endl
		<< "Rotation matrix: " << "\n" << relativePose_info.relativePose.rotation() << "\n\n"
		<< "Tranlation matrix:" << "\n" << relativePose_info.relativePose.translation() << "\n" << std::endl;


	//Triangulation
	SfM_Data Scene;
	Scene.views[0].reset(new View("", 0, 0,0,imageL.Width(), imageL.Height()));//initialize
	Scene.views[1].reset(new View("", 1,0,1, imageR.Width(), imageR.Height()));
//	std::cout << "111" << std::endl;

	Scene.intrinsics[0].reset(new Pinhole_Intrinsic(imageL.Width(), imageL.Height(), K(0, 0), K(0, 2), K(1, 2)));
//not used when use the same intrinsic (here)
	Scene.intrinsics[1].reset(new Pinhole_Intrinsic(imageL.Width(), imageL.Height(), K(0, 0), K(0, 2), K(1, 2)));

	// Setup poses camera data
	const Pose3 pose0 = Scene.poses[Scene.views[0]->id_pose] = Pose3(Mat3::Identity(), Vec3::Zero());
	const Pose3 pose1 = Scene.poses[Scene.views[1]->id_pose] = relativePose_info.relativePose;
//	std::cout << "222" << std::endl;

	// projection matrix 
	const Mat34 P1 = Scene.intrinsics[Scene.views[0]->id_intrinsic]->get_projective_equivalent(pose0);
	const Mat34 P2 = Scene.intrinsics[Scene.views[1]->id_intrinsic]->get_projective_equivalent(pose1);
	Landmarks & landmarks = Scene.structure;
//	std::cout << "success" << std::endl;

	for (size_t i = 0; i < relativePose_info.vec_inliers.size(); ++i) {
		const SIOPointFeature & LL = regionsL->Features()[vec_PutativeMatches[relativePose_info.vec_inliers[i]].i_];
		const SIOPointFeature & RR = regionsR->Features()[vec_PutativeMatches[relativePose_info.vec_inliers[i]].j_];
		// Point triangulation
		Vec3 X;
		TriangulateDLT(P1, LL.coords().cast<double>(), P2, RR.coords().cast<double>(), &X);
		// Reject point that is behind the camera
		if (pose0.depth(X) < 0 && pose1.depth(X) < 0)
			continue;
		// Add a new landmark (3D point with it's 2d observations)
		landmarks[i].obs[Scene.views[0]->id_view] = Observation(LL.coords().cast<double>(), vec_PutativeMatches[relativePose_info.vec_inliers[i]].i_);
		landmarks[i].obs[Scene.views[1]->id_view] = Observation(RR.coords().cast<double>(), vec_PutativeMatches[relativePose_info.vec_inliers[i]].j_);
		landmarks[i].X = X;
	}
	Save(Scene, "Scene.ply", ESfM_Data(ALL));
//	std::cout << "success" << std::endl;
}
return EXIT_SUCCESS;
}
  


