# generated from catkin/cmake/template/pkg.context.pc.in
CATKIN_PACKAGE_PREFIX = ""
PROJECT_PKG_CONFIG_INCLUDE_DIRS = "${prefix}/include;/usr/include/eigen3;/usr/local/include/pcl-1.10;/usr/include;/usr/include/ni;/usr/include/openni2;/usr/include/vtk-6.3;/usr/include/freetype2;/usr/include/x86_64-linux-gnu".split(';') if "${prefix}/include;/usr/include/eigen3;/usr/local/include/pcl-1.10;/usr/include;/usr/include/ni;/usr/include/openni2;/usr/include/vtk-6.3;/usr/include/freetype2;/usr/include/x86_64-linux-gnu" != "" else []
PROJECT_CATKIN_DEPENDS = "pcl_msgs;roscpp;sensor_msgs;std_msgs".replace(';', ' ')
PKG_CONFIG_LIBRARIES_WITH_PREFIX = "/usr/local/lib/libpcl_common.so;/usr/local/lib/libpcl_octree.so;/usr/local/lib/libpcl_io.so;/usr/lib/x86_64-linux-gnu/libboost_system.so;/usr/lib/x86_64-linux-gnu/libboost_filesystem.so;/usr/lib/x86_64-linux-gnu/libboost_date_time.so;/usr/lib/x86_64-linux-gnu/libboost_iostreams.so;/usr/lib/x86_64-linux-gnu/libboost_serialization.so;/usr/lib/x86_64-linux-gnu/libboost_regex.so;/usr/lib/libOpenNI.so;/usr/lib/libOpenNI2.so;/usr/lib/x86_64-linux-gnu/libvtkChartsCore-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkCommonColor-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkCommonDataModel-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkCommonMath-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkCommonCore-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtksys-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkCommonMisc-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkCommonSystem-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkCommonTransforms-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkInfovisCore-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkFiltersExtraction-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkCommonExecutionModel-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkFiltersCore-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkFiltersGeneral-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkCommonComputationalGeometry-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkFiltersStatistics-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkImagingFourier-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkImagingCore-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkalglib-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkRenderingContext2D-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkRenderingCore-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkFiltersGeometry-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkFiltersSources-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libfreetype.so;/usr/lib/x86_64-linux-gnu/libz.so;/usr/lib/x86_64-linux-gnu/libvtkftgl-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkFiltersModeling-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkImagingSources-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkInteractionWidgets-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkFiltersHybrid-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkImagingGeneral-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkImagingHybrid-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkIOImage-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkDICOMParser-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkIOCore-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkmetaio-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libjpeg.so;/usr/lib/x86_64-linux-gnu/libpng.so;/usr/lib/x86_64-linux-gnu/libtiff.so;/usr/lib/x86_64-linux-gnu/libvtkRenderingAnnotation-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkImagingColor-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkRenderingVolume-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkIOGeometry-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkIOLegacy-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkIOPLY-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkRenderingLOD-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkViewsContext2D-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkViewsCore-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkRenderingContextOpenGL-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkRenderingQt-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkFiltersTexture-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkGUISupportQt-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkRenderingLabel-6.3.so.6.3.0".split(';') if "/usr/local/lib/libpcl_common.so;/usr/local/lib/libpcl_octree.so;/usr/local/lib/libpcl_io.so;/usr/lib/x86_64-linux-gnu/libboost_system.so;/usr/lib/x86_64-linux-gnu/libboost_filesystem.so;/usr/lib/x86_64-linux-gnu/libboost_date_time.so;/usr/lib/x86_64-linux-gnu/libboost_iostreams.so;/usr/lib/x86_64-linux-gnu/libboost_serialization.so;/usr/lib/x86_64-linux-gnu/libboost_regex.so;/usr/lib/libOpenNI.so;/usr/lib/libOpenNI2.so;/usr/lib/x86_64-linux-gnu/libvtkChartsCore-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkCommonColor-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkCommonDataModel-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkCommonMath-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkCommonCore-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtksys-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkCommonMisc-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkCommonSystem-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkCommonTransforms-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkInfovisCore-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkFiltersExtraction-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkCommonExecutionModel-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkFiltersCore-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkFiltersGeneral-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkCommonComputationalGeometry-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkFiltersStatistics-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkImagingFourier-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkImagingCore-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkalglib-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkRenderingContext2D-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkRenderingCore-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkFiltersGeometry-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkFiltersSources-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libfreetype.so;/usr/lib/x86_64-linux-gnu/libz.so;/usr/lib/x86_64-linux-gnu/libvtkftgl-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkFiltersModeling-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkImagingSources-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkInteractionWidgets-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkFiltersHybrid-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkImagingGeneral-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkImagingHybrid-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkIOImage-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkDICOMParser-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkIOCore-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkmetaio-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libjpeg.so;/usr/lib/x86_64-linux-gnu/libpng.so;/usr/lib/x86_64-linux-gnu/libtiff.so;/usr/lib/x86_64-linux-gnu/libvtkRenderingAnnotation-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkImagingColor-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkRenderingVolume-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkIOGeometry-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkIOLegacy-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkIOPLY-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkRenderingLOD-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkViewsContext2D-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkViewsCore-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkRenderingContextOpenGL-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkRenderingQt-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkFiltersTexture-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkGUISupportQt-6.3.so.6.3.0;/usr/lib/x86_64-linux-gnu/libvtkRenderingLabel-6.3.so.6.3.0" != "" else []
PROJECT_NAME = "pcl_conversions"
PROJECT_SPACE_DIR = "/root/catkin_ws/install"
PROJECT_VERSION = "1.7.1"