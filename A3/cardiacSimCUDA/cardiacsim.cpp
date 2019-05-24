mirrorkernel<<<num_blocks, block_size>>>(d_E_prev_1D, n, m ,WIDTH);
			cudaStreamSynchronize(0);
			//cudaMemcpy(E_prev_1D, d_E_prev_1D, Total_Bytes, cudaMemcpyDeviceToHost);
			switch (version){
				case 1:
					simV1(alpha, n, m,  dt, WIDTH, &time_elapsed, d_E_1D, d_E_prev_1D, d_R_1D);
					break;
				case 2:
					simV2(alpha, n, m,  dt, WIDTH, &time_elapsed, d_E_1D, d_E_prev_1D, d_R_1D);
					break;
				case 3:
					simV3(alpha, n, m,  dt, WIDTH, &time_elapsed, d_E_1D, d_E_prev_1D, d_R_1D);
					break;
				case 4:
					
					simV4(alpha, n, m,  dt, WIDTH, &time_elapsed, d_E_1D, d_E_prev_1D, d_R_1D);
					break;
				// case 5:	
					
				// 	break;
				case 0:

				cout<<"\n Implement the Serial Version"<<endl;		
					break;
				default:
					cout<<"\nPlease Enter the Correct version"<<endl;
					return 0;
					
			}