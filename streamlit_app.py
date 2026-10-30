# ==============================================================================
# PAGE 4: ADMIN GOVERNANCE & TELEMETRY
# ==============================================================================
elif page == "🛡️ Admin Audit & Telemetry":
    st.header("🛡️ Enterprise Client Governance & Audit Portal")
    conn = get_db_connection()
    tab_manage, tab_client_inspect, tab_logs = st.tabs([
        "👥 Client Provisioning & Management", 
        "🔍 Inspect & Delete Client Models", 
        "📍 Access Telemetry"
    ])

    with tab_manage:
        st.subheader("Create Client Account")
        c_u1, c_u2, c_u3 = st.columns([2, 2, 1])
        new_client_user = c_u1.text_input("Username", key="admin_add_u")
        new_client_pass = c_u2.text_input("Password", type="password", key="admin_add_p")
        if c_u3.button("Create Account", type="primary") and new_client_user and new_client_pass:
            ok, msg = create_client_user(new_client_user, new_client_pass)
            st.success(msg) if ok else st.error(msg)
            if ok:
                st.rerun()

        st.divider()
        st.subheader("Manage Existing Client Accounts")
        users_df = pd.read_sql_query("SELECT username, role FROM users ORDER BY role ASC, username ASC", conn)
        
        c_list1, c_list2 = st.columns([3, 2])
        with c_list1:
            st.markdown("**Registered Users List**")
            st.dataframe(users_df, use_container_width=True)

        non_admin_users = [u for u in users_df["username"].tolist() if u != "admin"]
        with c_list2:
            st.markdown("**Account Actions**")
            if non_admin_users:
                selected_user = st.selectbox("Select Client to Manage:", options=non_admin_users, key="admin_sel_client")
                
                new_pwd = st.text_input("New Password", type="password", key="admin_pwd_reset_in")
                if st.button("🔑 Reset Client Password"):
                    if new_pwd.strip():
                        reset_client_password(selected_user, new_pwd)
                        st.success(f"Password reset for `{selected_user}`.")
                    else:
                        st.warning("Please enter a new password.")
                
                st.write("")
                confirm_del = st.checkbox(f"Confirm deletion of `{selected_user}` and all associated models", key="admin_del_confirm_box")
                if st.button("🗑️ Delete Client Account", type="secondary", disabled=not confirm_del):
                    c_models = conn.execute("SELECT model_path FROM protected_models WHERE username = ?", (selected_user,)).fetchall()
                    for m_row in c_models:
                        if m_row[0] and os.path.exists(m_row[0]):
                            try:
                                os.remove(m_row[0])
                            except Exception:
                                pass
                    delete_client_user(selected_user)
                    st.success(f"Client `{selected_user}` and all associated records permanently removed.")
                    st.rerun()
            else:
                st.info("No non-admin client accounts present.")

    with tab_client_inspect:
        st.subheader("Inspect & Delete Client Vault Models")
        users_df = pd.read_sql_query("SELECT username, role FROM users ORDER BY role ASC, username ASC", conn)
        all_clients = [u for u in users_df["username"].tolist() if u != "admin"]
        if all_clients:
            chosen_inspect_user = st.selectbox("Select Client:", options=all_clients, key="admin_inspect_client_sel")
            
            # Must query model_path along with id and model_tag
            c_mods = pd.read_sql_query(
                "SELECT id, model_tag, training_rows, created_at, model_path FROM protected_models WHERE username = ? ORDER BY id DESC", 
                conn, 
                params=(chosen_inspect_user,)
            )
            
            if not c_mods.empty:
                st.dataframe(c_mods[["id", "model_tag", "training_rows", "created_at"]], use_container_width=True)
                
                st.divider()
                st.markdown("#### 🗑️ Delete Specific Model")
                del_col1, del_col2 = st.columns([3, 1])
                
                model_map = {
                    f"ID {row['id']} — Tag: {row['model_tag']} (Created: {row['created_at']})": (row['id'], row['model_path'])
                    for _, row in c_mods.iterrows()
                }
                
                chosen_label = del_col1.selectbox("Select Model Artifact to Delete:", options=list(model_map.keys()), key="del_model_select")
                
                if del_col2.button("Delete Model", type="primary"):
                    target_id, target_path = model_map[chosen_label]
                    
                    # Delete physical file from disk if it exists
                    if target_path and os.path.exists(target_path):
                        try:
                            os.remove(target_path)
                        except Exception:
                            pass
                    
                    # Delete entry from SQLite database
                    conn.execute("DELETE FROM protected_models WHERE id = ?", (target_id,))
                    conn.commit()
                    sync_file_to_github(DB_PATH, DB_PATH, f"Auto-sync: Deleted model ID {target_id}")
                    st.success(f"Deleted `{chosen_label}` successfully.")
                    st.rerun()
            else:
                st.info(f"No models saved by `{chosen_inspect_user}`.")
        else:
            st.info("No client accounts found.")

    with tab_logs:
        st.subheader("Global Geolocation Telemetry")
        access_df = pd.read_sql_query("SELECT username, login_time, ip_address, city, region, country FROM access_logs ORDER BY login_time DESC", conn)
        st.dataframe(access_df, use_container_width=True)

    conn.close()
