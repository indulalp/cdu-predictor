# ==============================================================
            # RCO QUALITY ESTIMATOR: DISTILLATE RECOVERY AT 360°C (vol%)
            # ==============================================================
            st.divider()
            st.subheader("🧪 RCO Bottom Quality (Distillate Slippage @ 360°C)")

            # Operating inputs
            cot_curr = float(input_data.get('cot_degC', stats['mean_cot']))
            steam_curr = float(input_data.get('stripping_steam_flow', stats['mean_steam']))
            fzp_curr = float(input_data.get('flash_zone_p_kgcm2', stats['mean_p']))
            api_curr = float(calculated_api)

            # Baselines from plant operating anchor
            cot_base = stats['mean_cot']        # ~347.8 °C
            steam_base = stats['mean_steam']    # ~17.2 t/h
            fzp_base = stats['mean_p']          # ~1.56 kg/cm2
            api_base = stats['mean_api']        # ~30.5 °API

            # Physical stripped-overlap model
            # Base anchor centered at 6.5 vol% (midpoint of 6.0 - 7.0 range)
            base_rco_rec = 6.50
            
            # Thermodynamic sensitivities
            d_steam = -0.18 * (steam_curr - steam_base)
            d_cot   = -0.05 * (cot_curr - cot_base)
            d_fzp   = +1.20 * (fzp_curr - fzp_base)
            d_api   = -0.08 * (api_curr - api_base)

            # Monthly Lab Calibration Offset from Sidebar
            lab_offset = st.sidebar.number_input(
                "Monthly RCO @360°C Lab Bias", 
                value=0.0, 
                step=0.1, 
                help="Calibrate monthly using the delta between your 3 lab test results and the model output."
            )

            predicted_rco_360_vol = np.clip(
                base_rco_rec + d_steam + d_cot + d_fzp + d_api + lab_offset, 
                4.5, 9.5
            )

            rq_col1, rq_col2, rq_col3 = st.columns(3)
            rq_col1.metric("Predicted RCO Rec @ 360°C", f"{predicted_rco_360_vol:.2f} vol%")
            
            if predicted_rco_360_vol <= 6.2:
                status_color = "🟢 Optimal Stripping"
            elif predicted_rco_360_vol <= 7.0:
                status_color = "🟡 Normal Operation"
            else:
                status_color = "🔴 High Gas Oil Slippage"
                
            rq_col2.metric("Separation Status", status_color)
            rq_col3.metric("Effective Stripping Steam", f"{steam_curr:.1f} t/h")
