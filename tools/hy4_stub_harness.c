#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_hy4_resident_decode_stage_firmware.h"

int main(int argc, char **argv)
{
	SparkFirmwareModuleConfiguration configuration;
	SparkFirmwareModuleHostServices services;
	SparkModelDriverFrame frame;
	void *state;
	SparkStatus status;
	(void)argv;
	if ( argc != 1 )
	{
		fprintf(stderr,"usage: hy4_stub_harness\n");
		return(2);
	}
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_FIRMWARE_MODULE_ABI_VERSION;
	configuration.descriptor_bytes = sizeof(configuration);
	configuration.model_id = "hy4-t1";
	configuration.model_revision = "t1";
	configuration.stage_name = "hy4";
	configuration.program_name = "resident_decode";
	configuration.operation_name = "initialize";
	memset(&services,0,sizeof(services));
	services.abi_version = SPARK_FIRMWARE_MODULE_HOST_SERVICES_ABI_VERSION;
	services.descriptor_bytes = sizeof(services);
	state = 0;
	status = SparkHy4ResidentDecodeStageInitialize(&configuration,
		&services,&state);
	if ( status != SPARK_STATUS_OK || state == 0 )
	{
		fprintf(stderr,"STUB-BOUNDARY initialize status=%d\n",(int)status);
		return(1);
	}
	fprintf(stderr,"STUB-BOUNDARY initialize status=0 OK\n");
	memset(&frame,0,sizeof(frame));
	status = SparkHy4ResidentDecodeStageExecute(state,&frame);
	fprintf(stderr,"STUB-BOUNDARY execute status=%d (%s)\n",(int)status,
		SparkStatusToString(status));
	if ( status != SPARK_STATUS_UNSUPPORTED )
	{
		fprintf(stderr,"STUB-BOUNDARY unexpected execute status (fail loud)\n");
		SparkHy4ResidentDecodeStageDestroy(state);
		return(1);
	}
	SparkHy4ResidentDecodeStageDestroy(state);
	fprintf(stderr,"STUB-BOUNDARY destroy done\n");
	return(0);
}
