#include "preProc.h"

#include "DeviceHandler.h"
#include "cudaUtils.h"

#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
#include "TextureRenderer.h"
#include "globals.h"

#include<D3D11SDKLayers.h>
#include "D3DDebugger.h"

#include "thrust\system\system_error.h"

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
				   LPSTR lpCmdLine, int nCmdShow)
{
	int wndWidth = PIC_WIDTH+16;
	int wndHeight = PIC_HEIGHT+39;
	DeviceHandler* deviceHandler = new DeviceHandler( hInstance, wndWidth, wndHeight);
	D3DDebugger d3dDbg(deviceHandler->getDevice());
	TextureRenderer* texRender = new TextureRenderer(deviceHandler, PIC_WIDTH, PIC_HEIGHT );

	MSG msg = {0};

	// Hard coded for now
	float dt = 1.0f/60.0f;

	while( msg.message != WM_QUIT ){
		if( PeekMessage( &msg, NULL, 0, 0, PM_REMOVE ) ){
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		} else {
			//timer.tick();
			//update( timer.getDt() );
			//renderer->draw();
			//renderer->setWindowTitle( timer.getDt() );

			try{
				if( DeviceHandler::g_spacePressed || false){
					texRender->update( dt );
					DeviceHandler::g_spacePressed = false;
				}
				texRender->draw();
			} catch( thrust::system_error e ){
				Utils::error( e.what() );
			}
			//exit(0);
		}
	}

	delete texRender;
	delete deviceHandler;

	d3dDbg.reportLiveDeviceObjects();

	return 0;

}